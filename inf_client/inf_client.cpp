#include "inf_client.h"
#include "msgpack_lite.h"

#include <zmq.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

// ---- cppzmq state (kept out of the public header) ---------------------------
struct InfClient::Impl {
    zmq::context_t ctx{1};
    zmq::socket_t  sock{ctx, zmq::socket_type::req};
};

InfClient::InfClient(const std::string& endpoint, int recv_timeout_ms) : impl_(new Impl) {
    // Bounded recv so a dead/hung server surfaces as an error instead of a hang.
    impl_->sock.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);
    impl_->sock.set(zmq::sockopt::linger, 0);
    impl_->sock.connect(endpoint);
}

InfClient::~InfClient() { delete impl_; }

namespace {

// One request/reply round trip. Sends `req` bytes, returns the parsed reply.
mpk::Value roundtrip(zmq::socket_t& sock, const std::vector<uint8_t>& req) {
    sock.send(zmq::buffer(req), zmq::send_flags::none);
    zmq::message_t reply;
    zmq::recv_result_t r = sock.recv(reply, zmq::recv_flags::none);
    if (!r) throw std::runtime_error("inf_server: recv timed out (server down or stuck?)");
    mpk::Reader rd(static_cast<const uint8_t*>(reply.data()), reply.size());
    return rd.parse();
}

// Throw if reply is not {"ok": true, ...}, surfacing the server's error string.
void checkOk(const mpk::Value& reply) {
    const mpk::Value* ok = reply.find("ok");
    if (ok && ok->asBool()) return;
    const mpk::Value* err = reply.find("error");
    throw std::runtime_error("inf_server error: " +
                             (err && err->type == mpk::Value::STR ? err->s : "unknown"));
}

// Decode an ndarray envelope {"__ndarray__":true,"shape":[..],"dtype":str,"data":bin}
// of int16 (dtype "<i2"/"|i2") into a flat host vector (host is little-endian).
std::vector<int16_t> decodeI16(const mpk::Value& env, int& h, int& w) {
    const mpk::Value* shape = env.find("shape");
    const mpk::Value* data  = env.find("data");
    if (!shape || shape->type != mpk::Value::ARR || shape->arr.size() != 2 ||
        !data || data->type != mpk::Value::BIN)
        throw std::runtime_error("inf_server: malformed int16 ndarray envelope");
    h = (int)shape->arr[0].asInt();
    w = (int)shape->arr[1].asInt();
    const size_t n = (size_t)h * w;
    if (data->data.size() != n * sizeof(int16_t))
        throw std::runtime_error("inf_server: int16 ndarray size mismatch");
    std::vector<int16_t> out(n);
    std::memcpy(out.data(), data->data.data(), n * sizeof(int16_t));  // LE == host
    return out;
}

std::vector<float> decodeF32(const mpk::Value& env) {
    const mpk::Value* data = env.find("data");
    if (!data || data->type != mpk::Value::BIN) return {};
    const size_t n = data->data.size() / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), data->data.data(), n * sizeof(float));
    return out;
}

} // namespace

bool InfClient::ping() {
    mpk::Packer p;
    p.mapHeader(1);
    p.str("cmd"); p.str("ping");
    mpk::Value reply = roundtrip(impl_->sock, p.buf);
    const mpk::Value* ok = reply.find("ok");
    return ok && ok->asBool();
}

int InfClient::setVocab(const std::vector<std::string>& thing,
                        const std::vector<std::string>& stuff) {
    // Mirror the server ordering exactly: sorted(thing) ++ sorted(stuff).
    std::vector<std::string> t = thing, s = stuff;
    std::sort(t.begin(), t.end());
    std::sort(s.begin(), s.end());
    vocab_.clear();
    vocab_.insert(vocab_.end(), t.begin(), t.end());
    vocab_.insert(vocab_.end(), s.begin(), s.end());

    mpk::Packer p;
    p.mapHeader(3);
    p.str("cmd");           p.str("set_vocab");
    p.str("thing_classes"); p.arrHeader((uint32_t)thing.size());
    for (const auto& c : thing) p.str(c);
    p.str("stuff_classes"); p.arrHeader((uint32_t)stuff.size());
    for (const auto& c : stuff) p.str(c);

    mpk::Value reply = roundtrip(impl_->sock, p.buf);
    checkOk(reply);
    const mpk::Value* n = reply.find("num_classes");
    return n ? (int)n->asInt() : (int)vocab_.size();
}

void InfClient::reset() {
    mpk::Packer p;
    p.mapHeader(1);
    p.str("cmd"); p.str("reset");
    mpk::Value reply = roundtrip(impl_->sock, p.buf);
    checkOk(reply);
}

FrameResult InfClient::frame(const uint8_t* rgb, int h, int w) {
    mpk::Packer p;
    p.mapHeader(2);
    p.str("cmd");   p.str("frame");
    p.str("image");
    p.mapHeader(4);                                  // ndarray envelope
    p.str("__ndarray__"); p.boolean(true);
    p.str("shape"); p.arrHeader(3); p.uint((uint64_t)h); p.uint((uint64_t)w); p.uint(3);
    p.str("dtype"); p.str("|u1");                    // numpy uint8 dtype.str
    p.str("data");  p.bin(rgb, (size_t)h * w * 3);

    mpk::Value reply = roundtrip(impl_->sock, p.buf);
    checkOk(reply);

    FrameResult out;
    const mpk::Value* lm = reply.find("label_map");
    const mpk::Value* im = reply.find("id_map");
    if (!lm) throw std::runtime_error("inf_server: frame reply missing label_map");
    out.label_map = decodeI16(*lm, out.h, out.w);
    if (im) { int ih, iw; out.id_map = decodeI16(*im, ih, iw); }

    const mpk::Value* insts = reply.find("instances");
    if (insts && insts->type == mpk::Value::ARR) {
        out.instances.reserve(insts->arr.size());
        for (const mpk::Value& iv : insts->arr) {
            InfInstance in;
            if (const mpk::Value* v = iv.find("id"))    in.id    = (int)v->asInt();
            if (const mpk::Value* v = iv.find("label")) in.label = (int)v->asInt();
            if (const mpk::Value* v = iv.find("score")) in.score = (float)v->asDouble();
            if (const mpk::Value* v = iv.find("embedding")) in.embedding = decodeF32(*v);
            out.instances.push_back(std::move(in));
        }
    }
    return out;
}

const std::string& InfClient::className(int label) const {
    if (label < 0 || label >= (int)vocab_.size()) return empty_;
    return vocab_[label];
}
