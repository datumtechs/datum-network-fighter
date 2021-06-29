#!/bin/bash
protoc --go_out=plugins=grpc:./ compute_svc.proto
protoc --grpc-gateway_out=logtostderr=true:./ compute_svc.proto
protoc --go_out=plugins=grpc:./ data_svc.proto
protoc --grpc-gateway_out=logtostderr=true:./ data_svc.proto
