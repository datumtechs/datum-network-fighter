package conf

import (
	"testing"
)

func TestProxy(t *testing.T) {
	conf := LoadSSLConfig("./ssl-conf.yml")
	t.Log(conf.Conf.GMSSL.CaCertFile)
}
