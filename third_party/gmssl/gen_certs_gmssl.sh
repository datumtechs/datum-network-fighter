#!/bin/bash
#set -x

scripts_path=$(cd $(dirname $0); pwd)
CNF_FILE=${1:-"${scripts_path}/ssl.ini"}
echo "************: ${CNF_FILE}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${scripts_path}/lib
OPENSSL_CMD=${scripts_path}/lib/gmtassl
#OPENSSL_CNF=""                      # Tassl
OPENSSL_CNF="-config ${CNF_FILE}"     # GmSSL
OPENSSL_DIR=${scripts_path}
DAYS=3650
CA_CURVE=SM2

CERTS_DIR=${scripts_path}/../../certs
KEYS_DIR=$CERTS_DIR
COMBO_DIR=$CERTS_DIR
rm -rf $CERTS_DIR/*.crt $CERTS_DIR/*.key $CERTS_DIR/*.pem
mkdir -p $CERTS_DIR
mkdir -p $KEYS_DIR
mkdir -p $COMBO_DIR

function gen_root() {
    if [ $# -lt 2 ]; then
        echo "usage: root subj"
        return
    fi
    echo "=========================================================================="
    ROOT_FILE=$1
    SUBJ_DN=$2
    $OPENSSL_CMD ecparam -name $CA_CURVE -out $CA_CURVE.pem
    $OPENSSL_CMD req $OPENSSL_CNF -nodes -subj "$SUBJ_DN" \
        -keyout $KEYS_DIR/$ROOT_FILE.key \
        -newkey ec:$CA_CURVE.pem -new \
        -out $CERTS_DIR/$ROOT_FILE.req

    $OPENSSL_CMD x509 -req -days $DAYS \
	-sm3 \
        -in $CERTS_DIR/$ROOT_FILE.req \
        -extfile ${CNF_FILE} \
        -extensions v3_ca \
        -signkey $KEYS_DIR/$ROOT_FILE.key \
        -out $CERTS_DIR/$ROOT_FILE.crt

    $OPENSSL_CMD x509 -in $CERTS_DIR/$ROOT_FILE.crt -text
    $OPENSSL_CMD x509 -in $CERTS_DIR/$ROOT_FILE.crt -issuer -subject >$COMBO_DIR/$ROOT_FILE.pem
    cat $KEYS_DIR/$ROOT_FILE.key >>$COMBO_DIR/$ROOT_FILE.pem

    $OPENSSL_CMD pkcs12 -export -out $COMBO_DIR/$ROOT_FILE.pfx \
        -in $CERTS_DIR/$ROOT_FILE.crt -inkey $CERTS_DIR/$ROOT_FILE.key \
        -passout pass:123456

    rm $CERTS_DIR/$ROOT_FILE.req
}

function gen_cert() {
    if [ $# -lt 4 ]; then
        echo "usage: [sig|enc] root middle subj"
        return
    fi
    REQ_EXT=v3_req
    if [ "$1" = "sig" ]; then
        REQ_EXT=v3_req
    elif [ "$1" = "enc" ]; then
        REQ_EXT=v3enc_req
    else
        echo "usage: [sig|enc] root middle subj"
        return
    fi

    echo "=========================================================================="
    ROOT_FILE=$2
    CERT_FILE=$3
    SUBJ_DN=$4
    $OPENSSL_CMD req $OPENSSL_CNF -nodes -subj "$SUBJ_DN" \
        -keyout $KEYS_DIR/$CERT_FILE.key \
        -newkey ec:$CA_CURVE.pem -new \
        -out $CERTS_DIR/$CERT_FILE.req

    $OPENSSL_CMD x509 -req -days $DAYS \
	-sm3 \
        -in $CERTS_DIR/$CERT_FILE.req \
        -CA $CERTS_DIR/$ROOT_FILE.crt \
        -CAkey $KEYS_DIR/$ROOT_FILE.key \
        -extfile ${CNF_FILE} \
        -extensions $REQ_EXT \
        -CAcreateserial \
        -out $CERTS_DIR/$CERT_FILE.crt

    $OPENSSL_CMD x509 -in $CERTS_DIR/$CERT_FILE.crt -text
    $OPENSSL_CMD x509 -in $CERTS_DIR/$CERT_FILE.crt -issuer -subject >$COMBO_DIR/$CERT_FILE.pem
    cat $KEYS_DIR/$CERT_FILE.key >>$COMBO_DIR/$CERT_FILE.pem

    $OPENSSL_CMD pkcs12 -export -out $COMBO_DIR/$CERT_FILE.pfx \
        -in $CERTS_DIR/$CERT_FILE.crt -inkey $CERTS_DIR/$CERT_FILE.key \
        -passout pass:123456

    rm $CERTS_DIR/$CERT_FILE.req
}

# generate root certifcates [usage: root subj]
gen_root CA "/C=CN/ST=GD/L=SZ/O=Metis/OU=PCS/CN=Root CA (SM2)"

# generate middle certifcates [usage: sig|enc root middle subj]
gen_cert sig CA SS "/C=CN/ST=GD/L=SZ/O=Metis/OU=PCS/CN=Server Sign (SM2)"
gen_cert enc CA SE "/C=CN/ST=GD/L=SZ/O=Metis/OU=PCS/CN=Server Enc (SM2)"
gen_cert sig CA CS "/C=CN/ST=GD/L=SZ/O=Metis/OU=PCS/CN=Client Sign (SM2)"
gen_cert enc CA CE "/C=CN/ST=GD/L=SZ/O=Metis/OU=PCS/CN=Client Enc (SM2)"

# remove pfx/pem at present, 11/20/2020
# rm -rf $CERTS_DIR/*.pem $CERTS_DIR/*.pfx $CERTS_DIR/*.srl
rm -rf $CERTS_DIR/*.pfx $CERTS_DIR/*.srl ./*.pem

# Show all files
ls -al $CERTS_DIR/*.key $CERTS_DIR/*.crt $CERTS_DIR/*.pem

# # Hint
# echo -e "\n\033[33m!!! ATTENTION !!!\033[0m"
# echo -e "\033[32mPlease merge [sign].crt, [sign].key, [enc].crt and [enc].key \
# into one [both].pfx on site: https://www.gmssl.cn/gmssl/index.jsp.\033[0m"
# echo -e ""
echo "generate certs success. certs path = ${CERTS_DIR}"
