## 证书及密钥存放
### gmssl证书及密钥
所需证书及密钥的模板:
```
根证书及密钥:
  CA.crt
  CA.key
  CA.pem

服务器加密证书及密钥:
  SE.crt
  SE.key
  SE.pem
服务器签名证书及密钥:
  SS.crt
  SS.key
  SS.pem

客户端加密证书及密钥:
  CE.crt
  CE.key
  CE.pem
客户端签名证书及密钥:
  CS.crt
  CS.key
  CS.pem
```

### 产生证书
有两种方式：
1. 使用由组织或企业提供的证书及密钥
2. 由提供的脚本来生成，如下
```
cd third_party/gmssl
vi ssl.ini
# 修改[alt_names]下的IP.1为本组织的IP地址。不修改就默认IP是127.0.0.1，只适用于单机测试。
bash gen_certs_gmssl.sh
```


### 配置证书
如果证书及密钥文件名发生变化，请改动对应配置文件的内容，改动位置如下所示：
+ data_svc or compute_svc:
```
  # config.yaml
  certs: 
    base_path: ../certs  # the path to certs
    root_cert: CA.crt
    server_sign_key: SS.key
    server_sign_cert: SS.crt
    server_enc_key: SE.key
    server_enc_cert: SE.crt
    client_sign_key: CS.key
    client_sign_cert: CS.crt
    client_enc_key: CE.key
    client_enc_cert: CE.crt
```

+ third_party/via_svc:
```
  # conf/ssl-conf.yaml
  gmssl:
    # CA cert
    caCertFile: ../../certs/CA.crt

    # VIA server certs
    viaSignCertFile: ../../certs/SS.crt
    viaSignKeyFile: ../../certs/SS.key
    viaEncryptCertFile: ../../certs/SE.crt
    viaEncryptKeyFile: ../../certs/SE.key

    # io client certs, only valuable while mode=two_way
    ioSignCertFile: ../../certs/CS.crt
    ioSignKeyFile: ../../certs/CS.key
    ioEncryptCertFile: ../../certs/CE.crt
    ioEncryptKeyFile: ../../certs/CE.key
```