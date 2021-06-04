#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

typedef std::function<void(int, const std::string &, int, const std::string &, void *)> error_handler;
typedef std::function<std::string(const std::string &, void *)> message_handler;

/// Channel interface definition
/// the functionality is sending a message to peer and
/// receiving a message from peer
class IChannel
{
public:
  virtual ~IChannel() = default;

  /**
   * @brief Set error callback for error handle.
   * @param error_cb error callback to handle error.
   * @note Should set callback from python to c++, Rosetta internal should not set error callback.
  */
  virtual void SetErrorHandler(error_handler error_cb) = 0;

  /**
   * @brief RecvMessage receive a message from message queue， for the target node (blocking for timeout microseconds, default waiting forever)
   * @param party_id target node id for message receiving.
   * @param id identity of a message, could be a task id or message id.
   * @param data buffer to receive a message.
   * @param timeout timeout to receive a message.
   * @return 
   *  return message length if receive a message successfully
   *  0 if peer is disconnected  
   *  -1 if it gets a exception or error
   * if SetMessageHandler is called, then RecvMessage is desired to be disabled.
  */
  virtual int RecvMessage(uint64_t party_id, const std::string &id, std::string &data, int64_t timeout = -1) = 0;

  /**
   * @brief SendMessage send a message to target node
   * @param party_id target node id for message receiving
   * @param id identity of a message, could be a task id or message id.
   * @param data buffer to receive a message
   * @return 
   *  return 0 if send a message successfully
   *  -1 if gets exceptions or error
  */
  virtual int SendMessage(uint64_t party_id, const std::string &id, const std::string &data) = 0;
};
