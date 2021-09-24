package conf

import (
	"fmt"
	"gopkg.in/yaml.v3"
	"io/ioutil"
	"log"
)

type Config struct {
	Conf *Conf `yaml:"conf"`
}

type Conf struct {
	Cipher string `yaml:"cipher"`
	Mode   string `yaml:"mode"`
	SSL    *SSL   `yaml:"ssl"`
	GMSSL  *GMSSL `yaml:"gmssl"`
}
type SSL struct {
	CaCertFile  string `yaml:"caCertFile"`
	ViaCertFile string `yaml:"viaCertFile"`
	ViaKeyFile  string `yaml:"viaKeyFile"`
	IoCertFile  string `yaml:"ioCertFile"`
	IoKeyFile   string `yaml:"ioKeyFile"`
}
type GMSSL struct {
	CaCertFile         string `yaml:"caCertFile"`
	ViaSignCertFile    string `yaml:"viaSignCertFile"`
	ViaSignKeyFile     string `yaml:"viaSignKeyFile"`
	ViaEncryptCertFile string `yaml:"viaEncryptCertFile"`
	ViaEncryptKeyFile  string `yaml:"viaEncryptKeyFile"`
	IoSignCertFile     string `yaml:"ioSignCertFile"`
	IoSignKeyFile      string `yaml:"ioSignKeyFile"`
	IoEncryptCertFile  string `yaml:"ioEncryptCertFile"`
	IoEncryptKeyFile   string `yaml:"ioEncryptKeyFile"`
}

func LoadSSLConfig(configFile string) *Config {
	buf, err := ioutil.ReadFile(configFile)
	if err != nil {
		panic(fmt.Errorf("load ssl-conf file error. %v", err))
	}

	c := &Config{}
	err = yaml.Unmarshal(buf, c)
	if err != nil {
		panic(fmt.Errorf("load ssl-conf file error. %v", err))
	}

	log.Printf("sslConfig.cipher: %s,  mode: %s", c.Conf.Cipher, c.Conf.Mode)
	return c
}
