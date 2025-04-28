# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
import urllib.request
import time
from botocore.exceptions import ClientError

# クライアントクラス
class LLMClient:
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')
    
    def health_check(self):
        req = urllib.request.Request(
            url=f"{self.api_url}/health",
            method='GET'
        )
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read().decode())

    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True):
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url=f"{self.api_url}/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        start_time = time.time()
        with urllib.request.urlopen(req) as res:
            result = json.loads(res.read().decode())
        result["total_request_time"] = time.time() - start_time
        return result

def lambda_handler(event, context):
    api_url = os.getenv("API_URL")
    if not api_url:
        raise ValueError("API_URL is not set")

    # クライアントの初期化
    client = LLMClient(api_url)

    try:
        # ヘルスチェック
        health = client.health_check()
        print("Health check: ", health)

        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        # ユーザー発言を履歴に追加
        conversation_history.append({"role": "user", "content": message})

        # 単一の質問
        print("Processing message:", message)
        result = client.generate(message)
        # アシスタントの応答を取得
        assistant_response = result.get("generated_text")
        if not assistant_response:
            raise Exception("No response content from the model")

        print(f"Response: {assistant_response}")
        print(f"Model processing time: {result['response_time']:.2f}s")
        print(f"Total request time: {result['total_request_time']:.2f}s")  

        # 会話履歴を使用
        messages = conversation_history.copy()

        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
