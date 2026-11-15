# SPDX-License-Identifier: MIT
"""Deterministic free-model router. Standard library only; no inference gateway."""
import copy
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def text(content):
    if isinstance(content, str):
        return content
    return '\n'.join(b['text'] for b in (content or []) if b.get('type') == 'text')


def to_openai(body, wire):
    if wire == 'openai':
        return copy.deepcopy(body)
    out = {k: copy.deepcopy(v) for k, v in body.items()
           if k not in ('system', 'messages', 'tools', 'stop_sequences', 'tool_choice')}
    messages = []
    if body.get('system'):
        messages.append({'role': 'system', 'content': text(body['system'])})
    for message in body.get('messages', []):
        content = message.get('content', '')
        if isinstance(content, str):
            messages.append(copy.deepcopy(message))
            continue
        blocks, calls = [], []
        for block in content:
            kind = block['type']
            if kind == 'text':
                blocks.append(copy.deepcopy(block))
            elif kind == 'image':
                src = block['source']
                url = src.get('url') if src['type'] == 'url' else 'data:%s;base64,%s' % (src['media_type'], src['data'])
                blocks.append({'type': 'image_url', 'image_url': {'url': url}})
            elif kind == 'tool_use':
                calls.append({'id': block['id'], 'type': 'function', 'function': {'name': block['name'], 'arguments': compact(block['input'])}})
            elif kind == 'tool_result':
                if blocks:
                    messages.append({'role': message['role'], 'content': blocks})
                    blocks = []
                messages.append({'role': 'tool', 'tool_call_id': block['tool_use_id'], 'content': copy.deepcopy(block.get('content', ''))})
        if blocks or calls:
            m = {'role': message['role'], 'content': blocks or None}
            if calls:
                m['tool_calls'] = calls
            messages.append(m)
    out['messages'] = messages
    if 'stop_sequences' in body:
        out['stop'] = body['stop_sequences']
    if 'tools' in body:
        out['tools'] = [{'type': 'function', 'function': {**{k: v for k, v in t.items() if k != 'input_schema'}, 'parameters': t.get('input_schema', {})}} for t in body['tools']]
    choice = body.get('tool_choice', {})
    if choice:
        out['tool_choice'] = {'type':'function','function':{'name':choice['name']}} if choice.get('type') == 'tool' else {'any':'required'}.get(choice.get('type'),choice.get('type'))
    return out


def rank_candidates(rows, requires):
    eligible = [r for r in rows if r.get('free') is True
                and all(not requires.get(k) or r.get(k) is True for k in ('tools','vision','json'))
                and (r.get('context_length') or 0) >= requires.get('min_context',0)]
    return sorted(eligible, key=lambda r: (-r.get('rating_milli',0),r['provider'],r['model']))


def from_openai(body, wire, model):
    if wire == 'openai':
        return {**copy.deepcopy(body), 'model': model}
    systems = [text(m.get('content')) for m in body.get('messages', []) if m['role'] in ('system', 'developer')]
    max_tokens = body.get('max_tokens', body.get('max_completion_tokens'))
    stops = body.get('stop', [])
    if isinstance(stops, str):
        stops = [stops]
    if wire == 'anthropic':
        out = {'model': model, 'messages': [], 'stream': bool(body.get('stream'))}
        if systems:
            out['system'] = '\n'.join(systems)
        if max_tokens is not None:
            out['max_tokens'] = max_tokens
        if stops:
            out['stop_sequences'] = stops
        for msg in body.get('messages', []):
            if msg['role'] in ('system','developer'):
                continue
            if msg['role'] == 'tool':
                out['messages'].append({'role':'user','content':[{'type':'tool_result','tool_use_id':msg['tool_call_id'],'content':msg.get('content','')}]})
                continue
            content = msg.get('content')
            blocks = []
            if isinstance(content, str):
                blocks = [{'type':'text','text':content}]
            elif isinstance(content, list):
                for b in content:
                    if b['type'] == 'text':
                        blocks.append(copy.deepcopy(b))
                    elif b['type'] == 'image_url':
                        url = b['image_url']['url']
                        if url.startswith('data:') and ';base64,' in url:
                            mime, data = url[5:].split(';base64,',1)
                            src = {'type':'base64','media_type':mime,'data':data}
                        else:
                            src = {'type':'url','url':url}
                        blocks.append({'type':'image','source':src})
            for call in msg.get('tool_calls', []):
                blocks.append({'type':'tool_use','id':call['id'],'name':call['function']['name'],'input':json.loads(call['function']['arguments'])})
            out['messages'].append({'role':msg['role'],'content':blocks})
        if 'tools' in body:
            out['tools'] = [{**{k:v for k,v in t['function'].items() if k != 'parameters'},'input_schema':t['function'].get('parameters',{})} for t in body['tools']]
        choice = body.get('tool_choice')
        if choice:
            out['tool_choice'] = {'type':'tool','name':choice['function']['name']} if isinstance(choice,dict) else {'type':{'required':'any'}.get(choice,choice)}
        for key in ('temperature','top_p'):
            if key in body:
                out[key] = body[key]
        return out
    out = {'contents': [], 'generationConfig': {}}
    if systems:
        out['systemInstruction'] = {'parts':[{'text':'\n'.join(systems)}]}
    config = out['generationConfig']
    if max_tokens is not None:
        config['maxOutputTokens'] = max_tokens
    if stops:
        config['stopSequences'] = stops
    for source,target in [('temperature','temperature'),('top_p','topP')]:
        if source in body:
            config[target] = body[source]
    fmt = body.get('response_format',{})
    if fmt.get('type') in ('json_object','json_schema'):
        config['responseMimeType'] = 'application/json'
        if fmt.get('type') == 'json_schema':
            config['responseJsonSchema'] = fmt['json_schema']['schema']
    names = {c['id']:c['function']['name'] for m in body.get('messages',[]) for c in m.get('tool_calls',[])}
    for msg in body.get('messages',[]):
        if msg['role'] in ('system','developer'):
            continue
        parts = []
        if msg['role'] == 'tool':
            content = msg.get('content','')
            try:
                value = json.loads(content) if isinstance(content,str) else content
            except ValueError:
                value = {'result':content}
            if not isinstance(value,dict):
                value = {'result':value}
            parts.append({'functionResponse':{'name':names.get(msg['tool_call_id'],msg['tool_call_id']),'response':value}})
        else:
            content = msg.get('content')
            blocks = [{'type':'text','text':content}] if isinstance(content,str) else (content or [])
            for b in blocks:
                if b['type'] == 'text':
                    parts.append({'text':b['text']})
                elif b['type'] == 'image_url':
                    url = b['image_url']['url']
                    if url.startswith('data:') and ';base64,' in url:
                        mime,data = url[5:].split(';base64,',1)
                        parts.append({'inlineData':{'mimeType':mime,'data':data}})
                    else:
                        parts.append({'fileData':{'fileUri':url}})
            for call in msg.get('tool_calls',[]):
                parts.append({'functionCall':{'name':call['function']['name'],'args':json.loads(call['function']['arguments'])}})
        out['contents'].append({'role':'model' if msg['role']=='assistant' else 'user','parts':parts})
    if body.get('tools'):
        out['tools'] = [{'functionDeclarations':[copy.deepcopy(t['function']) for t in body['tools']]}]
    return out


def sse_values(raw):
    for frame in raw.replace('\r\n','\n').split('\n\n'):
        data = '\n'.join(line[5:].lstrip(' ') for line in frame.split('\n') if line.startswith('data:'))
        if data and data != '[DONE]':
            value = json.loads(data)
            if value.get('type') == 'error' or 'error' in value:
                raise ValueError('upstream stream error')
            yield value


def decode_answer(wire, raw, streaming, model):
    values = list(sse_values(raw)) if streaming else [json.loads(raw)]
    if not values:
        raise ValueError('empty upstream response')
    if streaming:
        if wire == 'openai' and not any(line.strip() == 'data: [DONE]' for line in raw.splitlines()):
            raise ValueError('truncated OpenAI stream')
        if wire == 'anthropic' and values[-1].get('type') != 'message_stop':
            raise ValueError('truncated Anthropic stream')
        if wire == 'gemini' and not any(c.get('finishReason') for v in values for c in v.get('candidates',[])):
            raise ValueError('truncated Gemini stream')
    answer = {'id':'aggregate','model':model,'text':'','calls':[], 'usage':(0,0),'stop':'stop','stop_sequence':None}
    calls = {}
    for value in values:
        answer['id'] = value.get('id',answer['id'])
        if wire == 'openai':
            if value.get('usage'):
                u = value['usage']
                answer['usage'] = (u.get('prompt_tokens',0),u.get('completion_tokens',0))
            for choice in value.get('choices',[])[:1]:
                msg = choice.get('delta' if streaming else 'message',{})
                answer['text'] += text(msg.get('content'))
                for i,c in enumerate(msg.get('tool_calls',[])):
                    idx = c.get('index',i)
                    dst = calls.setdefault(idx,{'id':'','type':'function','function':{'name':'','arguments':''}})
                    if c.get('id'):
                        dst['id'] = c['id']
                    for k in ('name','arguments'):
                        dst['function'][k] += c.get('function',{}).get(k,'')
                if choice.get('finish_reason'):
                    answer['stop'] = choice['finish_reason']
        elif wire == 'anthropic':
            kind = value.get('type')
            if kind == 'message_start':
                msg = value['message']
                answer['id'] = msg['id']
                answer['usage'] = (msg.get('usage',{}).get('input_tokens',0),0)
            elif kind == 'message' or not streaming:
                answer['text'] = text(value.get('content'))
                answer['usage'] = (value['usage']['input_tokens'],value['usage']['output_tokens'])
                answer['stop'] = value['stop_reason']
                answer['stop_sequence'] = value.get('stop_sequence')
                for i,b in enumerate(value.get('content',[])):
                    if b['type']=='tool_use':
                        calls[i] = {'id':b['id'],'type':'function','function':{'name':b['name'],'arguments':compact(b['input'])}}
            elif kind == 'content_block_start':
                b = value['content_block']
                if b['type'] == 'tool_use':
                    calls[value['index']] = {'id':b['id'],'type':'function','function':{'name':b['name'],'arguments':''}}
                elif b['type']=='text':
                    answer['text'] += b.get('text','')
            elif kind == 'content_block_delta':
                d = value['delta']
                if d['type']=='text_delta':
                    answer['text'] += d['text']
                elif d['type']=='input_json_delta':
                    calls[value['index']]['function']['arguments'] += d['partial_json']
            elif kind == 'message_delta':
                answer['stop'] = value['delta'].get('stop_reason',answer['stop'])
                answer['stop_sequence'] = value['delta'].get('stop_sequence')
                answer['usage'] = (answer['usage'][0],value.get('usage',{}).get('output_tokens',0))
        else:
            u = value.get('usageMetadata')
            if u:
                answer['usage'] = (u.get('promptTokenCount',0),u.get('candidatesTokenCount',0))
            for candidate in value.get('candidates',[])[:1]:
                for b in candidate.get('content',{}).get('parts',[]):
                    answer['text'] += b.get('text','')
                    if 'functionCall' in b:
                        c = b['functionCall']
                        i = len(calls)
                        calls[i] = {'id':c.get('id','call_'+str(i)),'type':'function','function':{'name':c['name'],'arguments':compact(c.get('args',{}))}}
                if candidate.get('finishReason'):
                    answer['stop'] = candidate['finishReason']
    answer['calls'] = [calls[i] for i in sorted(calls)]
    answer['stop'] = {'end_turn':'stop','stop_sequence':'stop','STOP':'stop','max_tokens':'length','MAX_TOKENS':'length','tool_use':'tool_calls','refusal':'content_filter','SAFETY':'content_filter'}.get(answer['stop'],answer['stop'])
    if answer['calls'] and answer['stop']=='stop':
        answer['stop'] = 'tool_calls'
    return answer


def response_json(a, wire):
    inp,out = a['usage']
    if wire == 'openai':
        msg = {'role':'assistant','content':a['text'] or (None if a['calls'] else '')}
        if a['calls']:
            msg['tool_calls'] = a['calls']
        return {'id':a['id'],'object':'chat.completion','created':0,'model':a['model'],
                'choices':[{'index':0,'message':msg,'finish_reason':a['stop']}],
                'usage':{'prompt_tokens':inp,'completion_tokens':out,'total_tokens':inp+out}}
    blocks = [{'type':'text','text':a['text']}] if a['text'] else []
    blocks += [{'type':'tool_use','id':c['id'],'name':c['function']['name'],'input':json.loads(c['function']['arguments'] or '{}')} for c in a['calls']]
    return {'id':a['id'],'type':'message','role':'assistant','model':a['model'],'content':blocks,
            'stop_reason':{'stop':'end_turn','length':'max_tokens','tool_calls':'tool_use','content_filter':'refusal'}.get(a['stop'],a['stop']),
            'stop_sequence':a.get('stop_sequence'),'usage':{'input_tokens':inp,'output_tokens':out}}


def response_sse(a, wire):
    frames = []
    def emit(value, event=None):
        frames.append(('event: '+event+'\n' if event else '')+'data: '+compact(value)+'\n\n')
    result = response_json(a,wire)
    if wire == 'openai':
        base = {k:result[k] for k in ('id','created','model')}
        base['object'] = 'chat.completion.chunk'
        def chunk(delta, finish=None):
            emit({**base,'choices':[{'index':0,'delta':delta,'finish_reason':finish}]})
        chunk({'role':'assistant','content':''})
        for i in range(0,len(a['text']),128):
            chunk({'content':a['text'][i:i+128]})
        for i,call in enumerate(a['calls']):
            chunk({'tool_calls':[{'index':i,**call}]})
        chunk({},a['stop'])
        emit({**base,'choices':[],'usage':result['usage']})
        frames.append('data: [DONE]\n\n')
    else:
        def event(kind, **data):
            emit({'type':kind,**data},kind)
        start = {**result,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':a['usage'][0],'output_tokens':0}}
        event('message_start',message=start)
        for i,block in enumerate(result['content']):
            if block['type']=='text':
                event('content_block_start',index=i,content_block={'type':'text','text':''})
                for at in range(0,len(block['text']),128):
                    event('content_block_delta',index=i,delta={'type':'text_delta','text':block['text'][at:at+128]})
            else:
                event('content_block_start',index=i,content_block={**block,'input':{}})
                event('content_block_delta',index=i,delta={'type':'input_json_delta','partial_json':compact(block['input'])})
            event('content_block_stop',index=i)
        event('message_delta',delta={'stop_reason':result['stop_reason'],'stop_sequence':result['stop_sequence']},usage={'output_tokens':a['usage'][1]})
        event('message_stop')
    return ''.join(frames)


def http(url, body=None, headers=None):
    request = urllib.request.Request(url, None if body is None else compact(body).encode(),
                                     {'Content-Type':'application/json',**(headers or {})})
    try:
        with urllib.request.urlopen(request,timeout=2) as response:
            return response.status,response.read(8*1024*1024).decode()
    except urllib.error.HTTPError as error:
        return error.code,error.read(1024*1024).decode()


def documents(source):
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        pos = source.find('{',pos)
        if pos < 0:
            return
        try:
            obj,end = decoder.raw_decode(source,pos)
        except ValueError:
            pos += 1
            continue
        pos = end
        if isinstance(obj,dict) and isinstance(obj.get('schema'),str):
            yield obj


class Router:
    def __init__(self, env_url, prompt):
        self.env = env_url.rstrip('/')
        self.providers = {}
        self.rows = []
        self.manifest = []
        self.catalog = {'providers':[]}
        self.index = 0
        self.failures = {}
        self.cooldown = {}
        docs = list(documents(prompt))
        ratings = {}
        for doc in docs:
            if doc['schema'].endswith('/ratings/v1'):
                ratings = {(r['provider'],r['model']):r for r in doc['ratings']}
            elif doc['schema'].endswith('/workload/v1'):
                self.manifest = doc['manifest']
        for doc in docs:
            if not doc['schema'].endswith('/providers/v1'):
                continue
            for p in doc['providers']:
                base = p.get('path') or p.get('base_url') or p.get('base_path')
                p = {**p,'base':base}
                self.providers[p['id']] = p
                url = self.base(p) + ('/v1beta/models' if p['wire']=='gemini' else '/v1/models')
                status,raw = http(url,headers=self.headers(p))
                if status != 200:
                    continue
                listing = json.loads(raw)
                models = []
                for item in listing.get('models' if p['wire']=='gemini' else 'data',[]):
                    mid = item.get('id') or item.get('name','').removeprefix('models/')
                    rating = ratings.get((p['id'],mid),{})
                    free = item.get('free',rating.get('free',False)) is True
                    row = {**rating,**item,'provider':p['id'],'model':mid,'free':free}
                    row['context_length'] = item.get('context_length',item.get('inputTokenLimit',rating.get('context_length')))
                    self.rows.append(row)
                    models.append({'id':mid,'free':free,'context_length':row['context_length']})
                self.catalog['providers'].append({'id':p['id'],'base_url':base,'key_env':p.get('key_env',''),'wire':p['wire'],'models':sorted(models,key=lambda m:m['id'])})

    def base(self,p):
        return p['base'].rstrip('/') if p['base'].startswith(('http://','https://')) else self.env+'/'+p['base'].strip('/')

    def headers(self,p,version='2023-06-01'):
        headers = {'anthropic-version':version}
        key = os.environ.get(p.get('key_env','')) or p.get('key','')
        auth = p.get('auth','none')
        if auth == 'bearer':
            headers['Authorization'] = 'Bearer '+key
        elif auth != 'none':
            headers[auth] = key
        return headers

    def answer(self,body,wire,version):
        normalized = to_openai(body,wire)
        req = dict(self.manifest[self.index].get('requires',{})) if self.index < len(self.manifest) else {}
        self.index += 1
        req['tools'] = req.get('tools',False) or bool(normalized.get('tools'))
        req['vision'] = req.get('vision',False) or any(b.get('type')=='image_url' for m in normalized.get('messages',[]) if isinstance(m.get('content'),list) for b in m['content'])
        req['json'] = req.get('json',False) or normalized.get('response_format',{}).get('type') in ('json_object','json_schema')
        candidates = rank_candidates(self.rows,req)
        candidates = [r for r in candidates if not (normalized.get('response_format') and self.providers[r['provider']]['wire']=='anthropic')]
        candidates.sort(key=lambda r:(self.cooldown.get(r['provider'],0)>self.index,-r.get('rating_milli',0),r['provider'],r['model']))
        tried = set()
        for row in candidates:
            pid = row['provider']
            if pid in tried:
                continue
            tried.add(pid)
            p = self.providers[pid]
            upstream = from_openai(normalized,p['wire'],row['model'])
            streaming = bool(body.get('stream'))
            if p['wire']=='openai':
                route = '/v1/chat/completions'
                if streaming:
                    upstream['stream_options'] = {'include_usage':True}
            elif p['wire']=='anthropic':
                route = '/v1/messages'
            else:
                route = '/v1beta/models/'+urllib.parse.quote(row['model'],safe='')+(':'+('streamGenerateContent?alt=sse' if streaming else 'generateContent'))
            try:
                status,raw = http(self.base(p)+route,upstream,self.headers(p,version))
            except (OSError,ValueError):
                status,raw = 503,''
            if status == 200:
                self.failures[pid] = 0
                self.cooldown[pid] = 0
                return decode_answer(p['wire'],raw,streaming,row['model'])
            if status in (402,429) or status >= 500:
                count = self.failures.get(pid,0)+1
                self.failures[pid] = count
                self.cooldown[pid] = self.index + min(16,2**count)
                continue
            # Authentication/model errors are never retried on the same provider.
            self.cooldown[pid] = self.index + len(self.manifest)+1
        raise RuntimeError('no available free provider meets the request')


def run():
    prompt_file = os.environ.get('MAKEFUN_PROMPT_FILE')
    prompt = open(prompt_file,encoding='utf-8').read() if prompt_file else ''
    router = Router(os.environ.get('MAKEFUN_ENV_URL',''),prompt)
    if '--list' in sys.argv:
        print(compact(router.catalog))
        return
    if not router.env:
        raise RuntimeError('MAKEFUN_ENV_URL is required')
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def send(self,status,value,stream=False):
            raw = (value if stream else compact(value)).encode()
            self.send_response_only(status)
            self.send_header('Content-Type','text/event-stream' if stream else 'application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            if self.path != '/v1/models':
                return self.send(404,{'error':{'message':'unknown route'}})
            ids = sorted({m['id'] for p in router.catalog['providers'] for m in p['models'] if m['free']})
            if self.headers.get('x-api-key'):
                rows = [{'id':mid,'type':'model','display_name':mid,'created_at':'2026-01-01T00:00:00Z'} for mid in ids]
                self.send(200,{'data':rows,'has_more':False,'first_id':ids[0] if ids else None,'last_id':ids[-1] if ids else None})
            else:
                self.send(200,{'object':'list','data':[{'id':mid,'object':'model','created':0,'owned_by':'aggregator'} for mid in ids]})
        def do_POST(self):
            wire = 'anthropic' if self.path=='/v1/messages' else 'openai'
            try:
                if self.path not in ('/v1/messages','/v1/chat/completions'):
                    return self.send(404,{'error':{'message':'unknown route'}})
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 32*1024*1024:
                    return self.send(413,{'error':{'message':'invalid body length'}})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body,dict):
                    raise ValueError('expected object')
                with lock:
                    answer = router.answer(body,wire,self.headers.get('anthropic-version','2023-06-01'))
                stream = body.get('stream') is True
                self.send(200,response_sse(answer,wire) if stream else response_json(answer,wire),stream)
            except Exception as error:
                # Never include upstream bodies, credentials, or URLs in diagnostics.
                status = 400 if isinstance(error,(ValueError,KeyError,TypeError)) else 503
                self.send(status,{'type':'error','error':{'type':'invalid_request_error' if status==400 else 'overloaded_error','message':'request could not be served'}})
    server = ThreadingHTTPServer(('0.0.0.0',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    status,raw = http(router.env+'/register',{'protocol_version':1,'port':server.server_port,'catalog':router.catalog})
    if status != 200 or not json.loads(raw).get('ok'):
        raise RuntimeError('registration failed')
    # Waiting is not a routing input; no timestamp enters any request or output.
    tick = threading.Event()
    while not tick.wait(0.05):
        status,raw = http(router.env+'/work')
        if status == 200 and json.loads(raw).get('done'):
            break
    server.shutdown()
    server.server_close()


if __name__ == '__main__':
    run()
