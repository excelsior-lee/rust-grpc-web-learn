use async_trait::async_trait;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;
use tokio::runtime::Builder;

use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Server;

use rust_grpc::infer_server;
use rust_grpc::InferRequest;
use rust_grpc::InferResponse;

struct InferImpl {
    model: SavedModelBundle,
    graph: Graph,
}

impl InferImpl {
    // 加载模型
    fn load(path: String) -> InferImpl {
        let mut graph = Graph::new();
        let opts = SessionOptions::new();
        let model = SavedModelBundle::load(&opts, &["serve"], &mut graph, path).unwrap();

        InferImpl { model, graph }
    }

    // 解析模型
    fn infer_impl(
        &self,
        x: Tensor<f32>,
    ) -> Tensor<f32> {
        let op = self
            .graph
            .operation_by_name_required("serving_default_input_1")
            .unwrap();
        let mut step = SessionRunArgs::new();
        step.add_feed(&op, 0, &x);

        let output_op = self
            .graph
            .operation_by_name_required("StatefulPartitionedCall")
            .unwrap();

        step.add_target(&output_op);
        let output_t = step.request_fetch(&output_op, 0);
        self.model.session.run(&mut step).unwrap();
        step.fetch(output_t).unwrap()
    }
}

// grpc 请求接口封装
#[async_trait]
impl infer_server::Infer for InferImpl {
    async fn infer(
        &self,
        req: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        let req = req.into_inner();
        let x: Tensor<f32> = Tensor::new(&req.shape).with_values(&req.data).unwrap();
        let output = self.infer_impl(x);
        let reply = InferResponse {
            shape: output.dims().into(),
            data: output.to_vec(),
        };
        Ok(Response::new(reply))
    }
}

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();

    rt.block_on(async {
        let addr = "0.0.0.0:7999";
        println!("Listen on: {}", addr);
        let addr = addr.parse().unwrap();
        let infer = InferImpl::load("pyts/resnet50".into());
        let server = infer_server::InferServer::new(infer);

        Server::builder()
            .add_service(server)
            .serve(addr)
            .await
            .unwrap();
    });
}