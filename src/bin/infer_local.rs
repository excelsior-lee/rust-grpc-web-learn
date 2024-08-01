use std::fs;
use std::slice;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

fn main() {
    // 加载模型
    let export_dir = "pyts/resnet50";
    let mut graph = Graph::new();
    let opts = SessionOptions::new();
    let sm = SavedModelBundle::load(&opts, &["serve"], &mut graph, export_dir).unwrap();

    let op = graph
        .operation_by_name_required("serving_default_input_1")
        .unwrap();
    let request = read_request();
    let mut step = SessionRunArgs::new();
    step.add_feed(&op, 0, &request);

    // for op in graph.operation_iter() {
    //     println!(">>> {:?}", op.name());
    // }
    let output_op = graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();

    step.add_target(&output_op);
    let output_t = step.request_fetch(&output_op, 0);
    sm.session.run(&mut step).unwrap();
    let output: Tensor<f32> = step.fetch(output_t).unwrap();
    println!("{:?}", output);
}

fn read_request() -> Tensor<f32> {
    let data = fs::read("pyts/request").unwrap();
    let data =
        unsafe { slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4 as usize) };

    Tensor::new(&[1, 224, 224, 3]).with_values(&data).unwrap()
}