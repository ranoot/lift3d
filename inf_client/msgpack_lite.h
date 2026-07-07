#pragma once
// Minimal, dependency-free msgpack for the inf_server IPC protocol.
//
// We control both ends and the schema is tiny (see inf_server/protocol.py), so a
// full msgpack library (with its optional-boost build) is overkill. This covers
// exactly the subset on the wire: maps/arrays/str/bin/bool/ints/floats. All
// multi-byte scalars are big-endian per the msgpack spec; the payloads inside the
// ndarray "data" bin stay little-endian (raw numpy bytes), decoded by the caller.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mpk {

// ---- writer -----------------------------------------------------------------
class Packer {
public:
    std::vector<uint8_t> buf;

    void mapHeader(uint32_t n)  { header(0x80, 0xde, 0xdf, n, /*fixmax=*/15); }
    void arrHeader(uint32_t n)  { header(0x90, 0xdc, 0xdd, n, /*fixmax=*/15); }

    void boolean(bool v) { buf.push_back(v ? 0xc3 : 0xc2); }

    void str(const std::string& s) {
        const uint32_t n = (uint32_t)s.size();
        if (n < 32)          buf.push_back(0xa0 | (uint8_t)n);
        else if (n < 256)  { buf.push_back(0xd9); u8((uint8_t)n); }
        else if (n < 65536){ buf.push_back(0xda); be(n, 2); }
        else               { buf.push_back(0xdb); be(n, 4); }
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void bin(const uint8_t* p, size_t n) {
        if (n < 256)        { buf.push_back(0xc4); u8((uint8_t)n); }
        else if (n < 65536) { buf.push_back(0xc5); be((uint32_t)n, 2); }
        else                { buf.push_back(0xc6); be((uint32_t)n, 4); }
        buf.insert(buf.end(), p, p + n);
    }

    void uint(uint64_t v) {
        if (v < 128) { buf.push_back((uint8_t)v); return; }         // positive fixint
        if (v < 256)        { buf.push_back(0xcc); u8((uint8_t)v); }
        else if (v < 65536) { buf.push_back(0xcd); be((uint32_t)v, 2); }
        else if (v <= 0xffffffffull) { buf.push_back(0xce); be((uint32_t)v, 4); }
        else { buf.push_back(0xcf); be64(v); }
    }

private:
    void u8(uint8_t v) { buf.push_back(v); }
    void be(uint32_t v, int bytes) {                     // big-endian, low `bytes`
        for (int i = bytes - 1; i >= 0; --i) buf.push_back((uint8_t)(v >> (8 * i)));
    }
    void be64(uint64_t v) { for (int i = 7; i >= 0; --i) buf.push_back((uint8_t)(v >> (8 * i))); }
    void header(uint8_t fix, uint8_t c16, uint8_t c32, uint32_t n, uint32_t fixmax) {
        if (n <= fixmax)      buf.push_back(fix | (uint8_t)n);
        else if (n < 65536) { buf.push_back(c16); be(n, 2); }
        else                { buf.push_back(c32); be(n, 4); }
    }
};

// ---- reader (parses into a tagged value tree) -------------------------------
struct Value {
    enum Type { NIL, BOOL, INT, UINT, FLOAT, STR, BIN, ARR, MAP } type = NIL;
    bool     b = false;
    int64_t  i = 0;
    uint64_t u = 0;
    double   d = 0;
    std::string          s;
    std::vector<uint8_t> data;                       // BIN
    std::vector<Value>   arr;                         // ARR
    std::vector<std::pair<Value, Value>> map;         // MAP

    const Value* find(const std::string& key) const {  // MAP lookup by str key
        if (type != MAP) return nullptr;
        for (auto& kv : map)
            if (kv.first.type == STR && kv.first.s == key) return &kv.second;
        return nullptr;
    }
    int64_t asInt() const {
        switch (type) { case INT: return i; case UINT: return (int64_t)u;
                        case BOOL: return b ? 1 : 0; case FLOAT: return (int64_t)d;
                        default: return 0; }
    }
    double asDouble() const {
        switch (type) { case FLOAT: return d; case INT: return (double)i;
                        case UINT: return (double)u; default: return 0.0; }
    }
    bool asBool() const { return type == BOOL ? b : asInt() != 0; }
};

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
    Value parse() { return readValue(); }

private:
    const uint8_t* p_;
    const uint8_t* end_;

    uint8_t byte() { if (p_ >= end_) throw std::runtime_error("msgpack: truncated"); return *p_++; }
    uint64_t be(int bytes) {
        uint64_t v = 0;
        for (int i = 0; i < bytes; ++i) v = (v << 8) | byte();
        return v;
    }
    std::string readStr(size_t n) {
        if (p_ + n > end_) throw std::runtime_error("msgpack: str overrun");
        std::string s((const char*)p_, n); p_ += n; return s;
    }
    std::vector<uint8_t> readBin(size_t n) {
        if (p_ + n > end_) throw std::runtime_error("msgpack: bin overrun");
        std::vector<uint8_t> v(p_, p_ + n); p_ += n; return v;
    }

    Value readValue() {
        uint8_t c = byte();
        Value v;
        if (c <= 0x7f)          { v.type = Value::UINT; v.u = c; return v; }          // pos fixint
        if (c >= 0xe0)          { v.type = Value::INT;  v.i = (int8_t)c; return v; }   // neg fixint
        if ((c & 0xf0) == 0x80) return readMap(c & 0x0f);                             // fixmap
        if ((c & 0xf0) == 0x90) return readArr(c & 0x0f);                             // fixarray
        if ((c & 0xe0) == 0xa0) { v.type = Value::STR; v.s = readStr(c & 0x1f); return v; } // fixstr
        switch (c) {
            case 0xc0: v.type = Value::NIL; return v;
            case 0xc2: v.type = Value::BOOL; v.b = false; return v;
            case 0xc3: v.type = Value::BOOL; v.b = true;  return v;
            case 0xc4: { size_t n = byte();     v.type = Value::BIN; v.data = readBin(n); return v; }
            case 0xc5: { size_t n = be(2);      v.type = Value::BIN; v.data = readBin(n); return v; }
            case 0xc6: { size_t n = be(4);      v.type = Value::BIN; v.data = readBin(n); return v; }
            case 0xca: { uint32_t r = (uint32_t)be(4); float f; std::memcpy(&f, &r, 4);
                         v.type = Value::FLOAT; v.d = f; return v; }
            case 0xcb: { uint64_t r = be(8); double dd; std::memcpy(&dd, &r, 8);
                         v.type = Value::FLOAT; v.d = dd; return v; }
            case 0xcc: v.type = Value::UINT; v.u = byte();  return v;
            case 0xcd: v.type = Value::UINT; v.u = be(2);   return v;
            case 0xce: v.type = Value::UINT; v.u = be(4);   return v;
            case 0xcf: v.type = Value::UINT; v.u = be(8);   return v;
            case 0xd0: v.type = Value::INT;  v.i = (int8_t)byte(); return v;
            case 0xd1: v.type = Value::INT;  v.i = (int16_t)be(2); return v;
            case 0xd2: v.type = Value::INT;  v.i = (int32_t)be(4); return v;
            case 0xd3: v.type = Value::INT;  v.i = (int64_t)be(8); return v;
            case 0xd9: { size_t n = byte(); v.type = Value::STR; v.s = readStr(n); return v; }
            case 0xda: { size_t n = be(2);  v.type = Value::STR; v.s = readStr(n); return v; }
            case 0xdb: { size_t n = be(4);  v.type = Value::STR; v.s = readStr(n); return v; }
            case 0xdc: return readArr(be(2));
            case 0xdd: return readArr(be(4));
            case 0xde: return readMap(be(2));
            case 0xdf: return readMap(be(4));
            default: throw std::runtime_error("msgpack: bad byte 0x" + std::to_string((int)c));
        }
    }
    Value readArr(size_t n) {
        Value v; v.type = Value::ARR; v.arr.reserve(n);
        for (size_t k = 0; k < n; ++k) v.arr.push_back(readValue());
        return v;
    }
    Value readMap(size_t n) {
        Value v; v.type = Value::MAP; v.map.reserve(n);
        for (size_t k = 0; k < n; ++k) { Value key = readValue(); Value val = readValue();
                                         v.map.emplace_back(std::move(key), std::move(val)); }
        return v;
    }
};

} // namespace mpk
