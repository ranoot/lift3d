#include "feat_client.h"
#include "msgpack_lite.h"

#include <zmq.hpp>

#include <cstring>
#include <stdexcept>

// ---- cppzmq state (kept out of the public header) ---------------------------
struct FeatClient::Impl {
    zmq::context_t ctx{1};
    zmq::socket_t  sock{ctx, zmq::socket_type::req};
};

FeatClient::FeatClient(const std::string& endpoint, int recv_timeout_ms) : impl_(new Impl) {
    // Bounded recv so a dead/hung server surfaces as an error instead of a hang.
    impl_->sock.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);
    impl_->sock.set(zmq::sockopt::linger, 0);
    impl_->sock.connect(endpoint);
}

FeatClient::~FeatClient() { delete impl_; }

namespace {

// One request/reply round trip. Sends `req` bytes, returns the parsed reply.
mpk::Value roundtrip(zmq::socket_t& sock, const std::vector<uint8_t>& req) {
    sock.send(zmq::buffer(req), zmq::send_flags::none);
    zmq::message_t reply;
    zmq::recv_result_t r = sock.recv(reply, zmq::recv_flags::none);
    if (!r) throw std::runtime_error("mask3d_feat: recv timed out (server down or stuck?)");
    mpk::Reader rd(static_cast<const uint8_t*>(reply.data()), reply.size());
    return rd.parse();
}

// Throw if reply is not {"ok": true, ...}, surfacing the server's error string.
void checkOk(const mpk::Value& reply) {
    const mpk::Value* ok = reply.find("ok");
    if (ok && ok->asBool()) return;
    const mpk::Value* err = reply.find("error");
    throw std::runtime_error("mask3d_feat error: " +
                             (err && err->type == mpk::Value::STR ? err->s : "unknown"));
}

// Pack a 2-D numpy ndarray envelope {"__ndarray__":true,"shape":[rows,cols],
// "dtype":..,"data":bin} from raw little-endian bytes (host is little-endian).
void packNdarray2D(mpk::Packer& p, uint64_t rows, uint64_t cols,
                   const char* dtype, const uint8_t* bytes, size_t nbytes) {
    p.mapHeader(4);
    p.str("__ndarray__"); p.boolean(true);
    p.str("shape"); p.arrHeader(2); p.uint(rows); p.uint(cols);
    p.str("dtype"); p.str(dtype);
    p.str("data");  p.bin(bytes, nbytes);
}

// Read a 2-D ndarray envelope's shape into rows/cols and return its raw bytes.
const std::vector<uint8_t>& ndarrayShape(const mpk::Value& env, int& rows, int& cols) {
    const mpk::Value* shape = env.find("shape");
    const mpk::Value* data  = env.find("data");
    if (!shape || shape->type != mpk::Value::ARR || shape->arr.size() != 2 ||
        !data || data->type != mpk::Value::BIN)
        throw std::runtime_error("mask3d_feat: malformed 2-D ndarray envelope");
    rows = (int)shape->arr[0].asInt();
    cols = (int)shape->arr[1].asInt();
    return data->data;
}

} // namespace

bool FeatClient::ping() {
    mpk::Packer p;
    p.mapHeader(1);
    p.str("cmd"); p.str("ping");
    mpk::Value reply = roundtrip(impl_->sock, p.buf);
    const mpk::Value* ok = reply.find("ok");
    return ok && ok->asBool();
}

FeatResult FeatClient::features(const std::vector<int32_t>& coords,
                                const std::vector<float>& feats3, int N) {
    FeatResult out;
    if (N <= 0) return out;
    if ((int)coords.size() != N * 3 || (int)feats3.size() != N * 3)
        throw std::runtime_error("mask3d_feat: coords/feats must be N*3");

    mpk::Packer p;
    p.mapHeader(3);
    p.str("cmd"); p.str("features");
    p.str("coords");
    packNdarray2D(p, (uint64_t)N, 3, "<i4",
                  reinterpret_cast<const uint8_t*>(coords.data()),
                  (size_t)N * 3 * sizeof(int32_t));
    p.str("feats");
    packNdarray2D(p, (uint64_t)N, 3, "<f4",
                  reinterpret_cast<const uint8_t*>(feats3.data()),
                  (size_t)N * 3 * sizeof(float));

    mpk::Value reply = roundtrip(impl_->sock, p.buf);
    checkOk(reply);

    const mpk::Value* cenv = reply.find("coords");
    const mpk::Value* fenv = reply.find("feats");
    if (!cenv || !fenv)
        throw std::runtime_error("mask3d_feat: features reply missing coords/feats");

    int crows, ccols, frows, fcols;
    const std::vector<uint8_t>& cbytes = ndarrayShape(*cenv, crows, ccols);
    const std::vector<uint8_t>& fbytes = ndarrayShape(*fenv, frows, fcols);
    if (crows != frows)
        throw std::runtime_error("mask3d_feat: coords/feats row count mismatch");
    if (cbytes.size() != (size_t)crows * ccols * sizeof(int32_t) ||
        fbytes.size() != (size_t)frows * fcols * sizeof(float))
        throw std::runtime_error("mask3d_feat: ndarray byte-size mismatch");

    out.M = frows;
    out.C = fcols;
    out.coords.resize((size_t)crows * ccols);
    out.feats.resize((size_t)frows * fcols);
    std::memcpy(out.coords.data(), cbytes.data(), cbytes.size());   // LE == host
    std::memcpy(out.feats.data(),  fbytes.data(), fbytes.size());
    return out;
}
