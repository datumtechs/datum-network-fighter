# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lib/types/types.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lib.types import header_pb2 as lib_dot_types_dot_header__pb2
from lib.types import metadata_pb2 as lib_dot_types_dot_metadata__pb2
from lib.types import resourcedata_pb2 as lib_dot_types_dot_resourcedata__pb2
from lib.types import taskdata_pb2 as lib_dot_types_dot_taskdata__pb2
from lib.types import identitydata_pb2 as lib_dot_types_dot_identitydata__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lib/types/types.proto',
  package='types',
  syntax='proto3',
  serialized_options=b'Z+github.com/RosettaFlow/Carrier-Go/lib/types',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15lib/types/types.proto\x12\x05types\x1a\x16lib/types/header.proto\x1a\x18lib/types/metadata.proto\x1a\x1clib/types/resourcedata.proto\x1a\x18lib/types/taskdata.proto\x1a\x1clib/types/identitydata.proto\"\xee\x01\n\tBlockData\x12\x1f\n\x06header\x18\x01 \x01(\x0b\x32\x0f.types.HeaderPb\x12#\n\x08metadata\x18\x02 \x03(\x0b\x32\x11.types.MetadataPB\x12\'\n\x0cresourcedata\x18\x03 \x03(\x0b\x32\x11.types.ResourcePB\x12\'\n\x0cidentitydata\x18\x04 \x03(\x0b\x32\x11.types.IdentityPB\x12\x1f\n\x08taskdata\x18\x05 \x03(\x0b\x32\r.types.TaskPB\x12\x12\n\nreceivedAt\x18\x06 \x01(\x04\x12\x14\n\x0creceivedFrom\x18\x07 \x01(\t\"\xb5\x01\n\x08\x42odyData\x12#\n\x08metadata\x18\x01 \x03(\x0b\x32\x11.types.MetadataPB\x12\'\n\x0cresourcedata\x18\x02 \x03(\x0b\x32\x11.types.ResourcePB\x12\'\n\x0cidentitydata\x18\x03 \x03(\x0b\x32\x11.types.IdentityPB\x12\x1f\n\x08taskdata\x18\x04 \x03(\x0b\x32\r.types.TaskPB\x12\x11\n\textraData\x18\x05 \x01(\x0c\"e\n\x0f\x44\x61taLookupEntry\x12\x11\n\tblockHash\x18\x01 \x01(\x0c\x12\x12\n\nblockIndex\x18\x02 \x01(\x04\x12\r\n\x05index\x18\x03 \x01(\x04\x12\x0e\n\x06nodeId\x18\x04 \x01(\t\x12\x0c\n\x04type\x18\x05 \x01(\tB-Z+github.com/RosettaFlow/Carrier-Go/lib/typesb\x06proto3'
  ,
  dependencies=[lib_dot_types_dot_header__pb2.DESCRIPTOR,lib_dot_types_dot_metadata__pb2.DESCRIPTOR,lib_dot_types_dot_resourcedata__pb2.DESCRIPTOR,lib_dot_types_dot_taskdata__pb2.DESCRIPTOR,lib_dot_types_dot_identitydata__pb2.DESCRIPTOR,])




_BLOCKDATA = _descriptor.Descriptor(
  name='BlockData',
  full_name='types.BlockData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='types.BlockData.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='types.BlockData.metadata', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='resourcedata', full_name='types.BlockData.resourcedata', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='identitydata', full_name='types.BlockData.identitydata', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='taskdata', full_name='types.BlockData.taskdata', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='receivedAt', full_name='types.BlockData.receivedAt', index=5,
      number=6, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='receivedFrom', full_name='types.BlockData.receivedFrom', index=6,
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
  serialized_start=169,
  serialized_end=407,
)


_BODYDATA = _descriptor.Descriptor(
  name='BodyData',
  full_name='types.BodyData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metadata', full_name='types.BodyData.metadata', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='resourcedata', full_name='types.BodyData.resourcedata', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='identitydata', full_name='types.BodyData.identitydata', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='taskdata', full_name='types.BodyData.taskdata', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='extraData', full_name='types.BodyData.extraData', index=4,
      number=5, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=410,
  serialized_end=591,
)


_DATALOOKUPENTRY = _descriptor.Descriptor(
  name='DataLookupEntry',
  full_name='types.DataLookupEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='blockHash', full_name='types.DataLookupEntry.blockHash', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='blockIndex', full_name='types.DataLookupEntry.blockIndex', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='index', full_name='types.DataLookupEntry.index', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='nodeId', full_name='types.DataLookupEntry.nodeId', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='types.DataLookupEntry.type', index=4,
      number=5, type=9, cpp_type=9, label=1,
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
  serialized_start=593,
  serialized_end=694,
)

_BLOCKDATA.fields_by_name['header'].message_type = lib_dot_types_dot_header__pb2._HEADERPB
_BLOCKDATA.fields_by_name['metadata'].message_type = lib_dot_types_dot_metadata__pb2._METADATAPB
_BLOCKDATA.fields_by_name['resourcedata'].message_type = lib_dot_types_dot_resourcedata__pb2._RESOURCEPB
_BLOCKDATA.fields_by_name['identitydata'].message_type = lib_dot_types_dot_identitydata__pb2._IDENTITYPB
_BLOCKDATA.fields_by_name['taskdata'].message_type = lib_dot_types_dot_taskdata__pb2._TASKPB
_BODYDATA.fields_by_name['metadata'].message_type = lib_dot_types_dot_metadata__pb2._METADATAPB
_BODYDATA.fields_by_name['resourcedata'].message_type = lib_dot_types_dot_resourcedata__pb2._RESOURCEPB
_BODYDATA.fields_by_name['identitydata'].message_type = lib_dot_types_dot_identitydata__pb2._IDENTITYPB
_BODYDATA.fields_by_name['taskdata'].message_type = lib_dot_types_dot_taskdata__pb2._TASKPB
DESCRIPTOR.message_types_by_name['BlockData'] = _BLOCKDATA
DESCRIPTOR.message_types_by_name['BodyData'] = _BODYDATA
DESCRIPTOR.message_types_by_name['DataLookupEntry'] = _DATALOOKUPENTRY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BlockData = _reflection.GeneratedProtocolMessageType('BlockData', (_message.Message,), {
  'DESCRIPTOR' : _BLOCKDATA,
  '__module__' : 'lib.types.types_pb2'
  # @@protoc_insertion_point(class_scope:types.BlockData)
  })
_sym_db.RegisterMessage(BlockData)

BodyData = _reflection.GeneratedProtocolMessageType('BodyData', (_message.Message,), {
  'DESCRIPTOR' : _BODYDATA,
  '__module__' : 'lib.types.types_pb2'
  # @@protoc_insertion_point(class_scope:types.BodyData)
  })
_sym_db.RegisterMessage(BodyData)

DataLookupEntry = _reflection.GeneratedProtocolMessageType('DataLookupEntry', (_message.Message,), {
  'DESCRIPTOR' : _DATALOOKUPENTRY,
  '__module__' : 'lib.types.types_pb2'
  # @@protoc_insertion_point(class_scope:types.DataLookupEntry)
  })
_sym_db.RegisterMessage(DataLookupEntry)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)