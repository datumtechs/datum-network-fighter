package common
import (
    "log"
    "flag"
    "github.com/larspensjo/config"
)
func ReadConfig(type_ string) map[string]string{
	//set config file std
	configFile := flag.String("configfile", "./common/config.ini", "General configuration file")
	info       := make(map[string]string)
	cfg, err := config.ReadDefault(*configFile)
	if err != nil {
		log.Fatalf("Fail to find", *configFile, err)
	}
	//set config file std End

	//Initialized info from the configuration
	if cfg.HasSection(type_) {
		section, err := cfg.SectionOptions(type_)
		if err == nil {
			for _, v := range section {
				options, err := cfg.String(type_, v)
				if err == nil {
					info[v] = options
				}
			}
		}
	}
	return info
	//Initialized info from the configuration END
}