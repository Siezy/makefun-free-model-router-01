# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import unittest
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SOURCE = Path(__file__).with_name('main.py')

class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.exists(), 'main.py implementation is missing')
        spec = importlib.util.spec_from_file_location('candidate', SOURCE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_anthropic_roundtrip_preserves_tools_and_images(self):
        request = {'model':'auto','system':[{'type':'text','text':'Keep exactly Ω'}],
            'max_tokens':77,'stop_sequences':['END'], 'messages':[
                {'role':'user','content':[{'type':'text','text':'Inspect this'},
                    {'type':'image','source':{'type':'base64','media_type':'image/png','data':'YWJj'}}]},
                {'role':'assistant','content':[{'type':'tool_use','id':'call_a','name':'lookup','input':{'x':3}}]},
                {'role':'user','content':[{'type':'tool_result','tool_use_id':'call_a','content':'found'}]}],
            'tools':[{'name':'lookup','description':'find','input_schema':{'type':'object','properties':{'x':{'type':'integer'}}}}]}
        normalized = self.m.to_openai(request, 'anthropic')
        self.assertEqual(normalized['messages'][0]['content'], 'Keep exactly Ω')
        self.assertEqual(normalized['messages'][1]['content'][1]['image_url']['url'], 'data:image/png;base64,YWJj')
        self.assertEqual(normalized['messages'][-1]['tool_call_id'],'call_a')
        result = self.m.from_openai(normalized, 'anthropic', 'chosen')
        self.assertEqual(result['tools'], request['tools'])
        self.assertEqual(result['messages'], request['messages'])
        self.assertEqual(result['max_tokens'],77)
        self.assertEqual(result['stop_sequences'],['END'])
        self.assertEqual(result['system'],'Keep exactly Ω')

    def test_gemini_preserves_request_fields(self):
        request = {'messages':[{'role':'system','content':'precise'}, {'role':'user','content':[
            {'type':'text','text':'two words'}, {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,eHl6'}}]}],
            'max_tokens':123,'stop':['STOP'],'response_format':{'type':'json_object'},
            'tools':[{'type':'function','function':{'name':'f','parameters':{'type':'object'}}}]}
        actual = self.m.from_openai(request,'gemini','m')
        self.assertEqual(actual['systemInstruction']['parts'],[{'text':'precise'}])
        self.assertEqual(actual['contents'][0]['parts'][1],{'inlineData':{'mimeType':'image/jpeg','data':'eHl6'}})
        self.assertEqual(actual['generationConfig'],{'maxOutputTokens':123,'stopSequences':['STOP'],'responseMimeType':'application/json'})
        self.assertEqual(actual['tools'][0]['functionDeclarations'][0]['name'],'f')

class ResponseTests(TranslationTests):
    def test_stream_tool_arguments_and_usage(self):
        chunks = [
            {'id':'answer','model':'m','choices':[{'delta':{'role':'assistant','tool_calls':[{'index':0,'id':'call_1','type':'function','function':{'name':'lookup','arguments':'{"x":'}}]},'finish_reason':None}]},
            {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'4}'}}]},'finish_reason':None}]},
            {'choices':[{'delta':{},'finish_reason':'tool_calls'}]},
            {'choices':[],'usage':{'prompt_tokens':19,'completion_tokens':7}}]
        raw = ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
        answer = self.m.decode_answer('openai',raw,True,'m')
        self.assertEqual(answer['calls'][0]['function']['arguments'],'{"x":4}')
        self.assertEqual(answer['usage'],(19,7))
        converted = self.m.response_json(answer,'anthropic')
        self.assertEqual(converted['content'][0]['input'],{'x':4})
        self.assertEqual(converted['stop_reason'],'tool_use')
        events = self.m.response_sse(answer,'anthropic')
        self.assertLess(events.index('content_block_stop'),events.index('message_delta'))
        self.assertIn('input_json_delta',events)

    def test_capability_routing_rejects_paid_and_incompatible(self):
        rows = [
            {'provider':'a','model':'paid','free':False,'rating_milli':2000,'tools':True},
            {'provider':'a','model':'weak','free':True,'rating_milli':1000,'tools':True},
            {'provider':'b','model':'strong','free':True,'rating_milli':1500,'tools':True},
            {'provider':'c','model':'text','free':True,'rating_milli':1900,'tools':False}]
        self.assertEqual([r['model'] for r in self.m.rank_candidates(rows,{'tools':True})],['strong','weak'])

class HttpTests(TranslationTests):
    def test_real_http_402_429_failover_and_anthropic_version(self):
        seen = []
        class Provider(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path,body,self.headers.get('anthropic-version')))
                code = 402 if self.path.startswith('/a/') else 429 if self.path.startswith('/b/') else 200
                value = {'error':{'message':'unavailable'}} if code != 200 else {
                    'id':'real-http-fixture','type':'message','role':'assistant','model':'m',
                    'content':[{'type':'text','text':'invented upstream response Ω'}],
                    'stop_reason':'max_tokens','usage':{'input_tokens':8,'output_tokens':3}}
                self.send_response(code)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())
        server = ThreadingHTTPServer(('127.0.0.1',0),Provider)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            r = self.m.Router('','')
            r.env = 'http://127.0.0.1:'+str(server.server_port)
            for i,pid in enumerate(('a','b','c')):
                r.providers[pid] = {'id':pid,'wire':'anthropic','base':'/'+pid,'auth':'none'}
                r.rows.append({'provider':pid,'model':'m','free':True,'rating_milli':100-i,'context_length':1000})
            answer = r.answer({'messages':[{'role':'user','content':'hello'}],'max_tokens':3},'openai','2024-01-01')
            self.assertEqual(answer['text'],'invented upstream response Ω')
            self.assertEqual(answer['stop'],'length')
            self.assertEqual(answer['usage'],(8,3))
            self.assertEqual([x[0] for x in seen],['/a/v1/messages','/b/v1/messages','/c/v1/messages'])
            self.assertTrue(all(x[2]=='2024-01-01' for x in seen))
        finally:
            server.shutdown()
            server.server_close()

    def test_gemini_stream_and_anthropic_stream_roundtrip(self):
        value = {'candidates':[{'content':{'parts':[{'text':'fragment Ω'}]},'finishReason':'MAX_TOKENS'}],
                 'usageMetadata':{'promptTokenCount':11,'candidatesTokenCount':6}}
        a = self.m.decode_answer('gemini','data: '+json.dumps(value)+'\n\n',True,'fixture')
        for wire in ('anthropic','openai'):
            b = self.m.decode_answer(wire,self.m.response_sse(a,wire),True,'fixture')
            self.assertEqual(b['text'],a['text'])
            self.assertEqual(b['usage'],a['usage'])
            self.assertEqual(b['stop'],'length')

    def test_missing_or_truncated_sse_is_rejected(self):
        with self.assertRaises(ValueError):
            self.m.decode_answer('openai','data: {"choices":[]}\n\n',True,'fixture')

if __name__ == '__main__':
    unittest.main()
