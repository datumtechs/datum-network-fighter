# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lib/types/identitydata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lib.common import base_pb2 as lib_dot_common_dot_base__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lib/types/identitydata.proto',
  package='types',
  syntax='proto3',
  serialized_options=b'Z+github.com/RosettaFlow/Carrier-Go/lib/types',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1clib/types/identitydata.proto\x12\x05types\x1a\x15lib/common/base.proto\"\xc5\x01\n\nIdentityPB\x12\x13\n\x0bidentity_id\x18\x01 \x01(\t\x12\x0f\n\x07node_id\x18\x02 \x01(\t\x12\x11\n\tnode_name\x18\x03 \x01(\t\x12\x0f\n\x07\x64\x61ta_id\x18\x04 \x01(\t\x12-\n\x0b\x64\x61ta_status\x18\x05 \x01(\x0e\x32\x18.api.protobuf.DataStatus\x12*\n\x06status\x18\x06 \x01(\x0e\x32\x1a.api.protobuf.CommonStatus\x12\x12\n\ncredential\x18\x07 \x01(\tB-Z+github.com/RosettaFlow/Carrier-Go/lib/typesb\x06proto3'
  ,
  dependencies=[lib_dot_common_dot_base__pb2.DESCRIPTOR,])




_IDENTITYPB = _descriptor.Descriptor(
  name='IdentityPB',
  full_name='types.IdentityPB',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='identity_id', full_name='types.IdentityPB.identity_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_id', full_name='types.IdentityPB.node_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_name', full_name='types.IdentityPB.node_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_id', full_name='types.IdentityPB.data_id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_status', full_name='types.IdentityPB.data_status', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status', full_name='types.IdentityPB.status', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='credential', full_name='types.IdentityPB.credential', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=260,
)

_IDENTITYPB.fields_by_name['data_status'].enum_type = lib_dot_common_dot_base__pb2._DATASTATUS
_IDENTITYPB.fields_by_name['status'].enum_type = lib_dot_common_dot_base__pb2._COMMONSTATUS
DESCRIPTOR.message_types_by_name['IdentityPB'] = _IDENTITYPB
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IdentityPB = _reflection.GeneratedProtocolMessageType('IdentityPB', (_message.Message,), {
  'DESCRIPTOR' : _IDENTITYPB,
  '__module__' : 'lib.types.identitydata_pb2'
  # @@protoc_insertion_point(class_scope:types.IdentityPB)
  })
_sym_db.RegisterMessage(IdentityPB)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
