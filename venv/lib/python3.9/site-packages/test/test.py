import unittest
import os.path
import json

from asposestorage.ApiClient import ApiClient
from asposestorage.StorageApi import StorageApi




class TestAsposeStorage(unittest.TestCase):

    def setUp(self):

        with open('setup.json') as json_file:
            data = json.load(json_file)

        self.apiClient = ApiClient(apiKey=str(data['app_key']),appSid=str(data['app_sid']),apiServer=str(data['product_uri']))
        self.storageApi = StorageApi(self.apiClient)

        self.output_path = str(data['output_location'])

    def testGetListFiles(self):
        response = self.storageApi.GetListFiles(Path='')
        self.assertEqual(response.status_code,200)
        self.assertEqual(dict, type(json.loads(response.content)))
        print response.content

    def testGetDiscUsage(self):
        response = self.storageApi.GetDiscUsage()
        self.assertEqual(response.status_code,200)
        self.assertEqual(dict, type(json.loads(response.content)))

    def testPutCreate(self):
        response = self.storageApi.PutCreate('testfile.txt','./data/testfile.txt')
        self.assertEqual(response.status_code,200)

    def testGetDownload(self):
        response = self.storageApi.GetDownload('testfile.txt')

        with open(self.output_path + 'testfile.txt', 'wb') as f:
            for chunk in response.iter_content():
                f.write(chunk)

        self.assertTrue(True, os.path.exists(self.output_path + 'testfile.txt'))

    def testGetIsExist(self):
        response = self.storageApi.GetIsExist('testfile.txt')
        self.assertEqual(response.status_code,200)
        json_response = json.loads(response.content)
        self.assertEqual(True,json_response['FileExist']['IsExist'])

    def testPutCreateFolder(self):
        response = self.storageApi.PutCreateFolder('mytestfolder')
        self.assertEqual(response.status_code,200)

    def testPostMoveFile(self):
        response = self.storageApi.PostMoveFile('testfile.txt','mytestfolder/testfile.txt')
        print response.content
        self.assertEqual(response.status_code,200)

    def testPostMoveFolder(self):
        response = self.storageApi.PostMoveFolder('mytestfolder','mytestfolder_new')
        self.assertEqual(response.status_code,200)

    def testPutCopy(self):
        response = self.storageApi.PutCopy('testfile.txt','new_testfile.txt')
        self.assertEqual(response.status_code,200)


    def testPutCopyFolder(self):
        response = self.storageApi.PutCopyFolder('mytestfolder','mytestfolder1')
        self.assertEqual(response.status_code,200)

    def testGetIsStorageExist(self):
        response = self.storageApi.GetIsStorageExist('Aspose123')
        json_response = json.loads(response.content)
        self.assertEqual(response.status_code,200)
        self.assertEqual(False,json_response['IsExist'])

    def testGetListFileVersions(self):
        response = self.storageApi.GetListFileVersions('testfile.txt')
        self.assertEqual(response.status_code,200)

    def testDeleteFolder(self):
        response = self.storageApi.DeleteFolder('mytestfolder')
        self.assertEqual(response.status_code,200)

    def testDeleteFile(self):
        response = self.storageApi.DeleteFile('testfile.txt')
        self.assertEqual(response.status_code,200)


