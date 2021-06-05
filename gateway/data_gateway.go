package main

import (
	"flag"
	"net/http"

	gw "gateway/data_svc"
	"gateway/common"
	"github.com/golang/glog"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)


func run() error {
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	info :=common.ReadConfig("DataSvc")
	mux := runtime.NewServeMux()
	opts := []grpc.DialOption{grpc.WithInsecure()}
	echoEndpoint := flag.String("echo_endpoint", info["rpc_host"]+":"+info["rpc_port"], "endpoint of YourService")
	err := gw.RegisterDataProviderHandlerFromEndpoint(ctx, mux, *echoEndpoint, opts)
	if err != nil {
		return err
	}

	return http.ListenAndServe(":"+info["http_port"], mux)
}

func main() {
	flag.Parse()
	defer glog.Flush()

	if err := run(); err != nil {
		glog.Fatal(err)
	}
}
