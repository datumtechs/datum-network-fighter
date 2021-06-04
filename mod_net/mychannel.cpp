#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>

#include "io_channel.h"

namespace py = pybind11;

using std::cout;
using std::endl;
using std::string;
using std::vector;

class MyChannelImpl : public IChannel
{
public:
    MyChannelImpl(const string &task_id, const vector<string> &peers) : task_id(task_id), peers(peers) {}
    ~MyChannelImpl() {}
    virtual void SetErrorHandler(error_handler error_cb)
    {
        err_cb = error_cb;
    }
    virtual int RecvMessage(uint64_t party_id, const string &id, string &data, int64_t timeout = -1)
    {
        if (party_id > 10)
        {
            err_cb(1, "task01", 2, "proc01", &task_id);
            return 0;
        }
        data = "matrix A";
        return 1;
    }
    virtual int SendMessage(uint64_t party_id, const string &id, const string &data)
    {
        cout << "task_id: " << task_id << ", party_id: " << party_id << ", data: " << data << endl;
        for (auto &&p : peers)
            cout << p << endl;
        return 2;
    }

private:
    string task_id;
    const vector<string> peers;
    error_handler err_cb;
};

PYBIND11_MODULE(mychannel, m)
{
    m.doc() = "pybind11 mychannel plugin";
    py::class_<MyChannelImpl>(m, "Channel")
        .def(py::init<const string &, const vector<string> &>())
        .def("set_err_cb", &MyChannelImpl::SetErrorHandler)
        .def("recv", &MyChannelImpl::RecvMessage)
        .def("send", &MyChannelImpl::SendMessage);
}
