import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from experiments.demo_server import make_handler


class Fake:
    def registry(self):return {'models':[{'id':'race','available':True}],'device':'cpu'}
    def predict(self,text,name):return {'echo':text,'model':name}

class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def explain(s,t,n,**kw):self.calls.append((t,n,kw));return {'schema':'test'}
        self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(Fake(),explain,b'NEW DOCUMENT VIEW',b'LEGACY READING VIEW',0))
        port=self.server.server_address[1]
        self.server.RequestHandlerClass=make_handler(Fake(),explain,b'NEW DOCUMENT VIEW',b'LEGACY READING VIEW',port)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url=f'http://127.0.0.1:{port}'
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,path,data=None,**headers):
        raw=json.dumps(data).encode() if data is not None else None
        req=urllib.request.Request(self.url+path,data=raw,headers={'Content-Type':'application/json',**headers})
        try:
            with urllib.request.urlopen(req,timeout=5) as r:return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    def test_pages_preserve_legacy(self):
        self.assertEqual(self.request('/')[1],b'NEW DOCUMENT VIEW');self.assertEqual(self.request('/reading')[1],b'LEGACY READING VIEW')
    def test_original_predict_contract(self):self.assertEqual(json.loads(self.request('/api/predict',{'text':'hello','model':'race'})[1]),{'echo':'hello','model':'race'})
    def test_explanation_route(self):
        self.assertEqual(self.request('/api/explain',{'text':'full','model':'race','unit_ids':['u1']})[0],200);self.assertEqual(self.calls[0][0],'full')
    def test_registry_capability(self):self.assertEqual(json.loads(self.request('/api/models')[1])['explanation_api'],'ee6405.document-explanation/v1')
    def test_foreign_origin(self):self.assertEqual(self.request('/api/explain',{},Origin='https://not-local.example')[0],403)
    def test_array_request(self):self.assertEqual(self.request('/api/explain',[])[0],400)
    def test_path_traversal(self):self.assertEqual(self.request('/../../etc/passwd')[0],404)
    def test_unknown_post(self):self.assertEqual(self.request('/api/other',{})[0],404)
if __name__=='__main__':unittest.main()
