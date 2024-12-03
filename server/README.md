## 语义通信服务器（server)

顶端文件：**`main.py`**

功能文件：**`./utils`**

### 接口文档

#### 处理图片

```json
method: POST
url: "/api/image/bake"
// 处理图片需要的参数
body:{
    param1: "",
    param2: "",
    ...
}
response:{
    message: string
}
```

#### 请求图片

```json
method: GET
url: "/api/image/request"
body:{
    // 暂定不需要参数
}
response:{
    images: Object{} // images[]  {name:"", src:""}
}
DB action: 查User表检验密码是否正确
```

### 安装

```shell
pip install -r requirements.txt
```

### 运行

```shell
python ./main.py
```

