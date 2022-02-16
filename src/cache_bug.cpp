#include <openvino/openvino.hpp>

int main(){
    ov::Core ieCore;
    ieCore.set_property({{CONFIG_KEY(CACHE_DIR), "./cache"}});
    auto model = ieCore.read_model("./model");
    ov::CompiledModel compiledModel = ieCore.compile_model(model, "GPU");
    model = ieCore.read_model("./model");
    compiledModel = ieCore.compile_model(model, "GPU");
}
