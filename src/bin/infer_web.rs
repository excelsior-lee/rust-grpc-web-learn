use rust_grpc::{
    infer_client::InferClient, process_client::ProcessClient, AfterProcessRequest,
    AfterProcessResponse, InferRequest, InferResponse, PreProcessRequest, PreProcessResponse,
};
use tokio::runtime::Builder;

use axum::{
    extract::{ContentLengthLimit, Extension, Multipart},
    response::{Html, IntoResponse},
    routing::get,
    AddExtensionLayer, Json, Router,
};
use std::iter::Iterator;
use tonic::transport::Channel;

use serde::Deserialize;
use serde::Serialize;

#[derive(Clone)]
struct Clients {
    infer_cli: InferClient<Channel>,
    process_cli: ProcessClient<Channel>,
}

fn main() {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        if std::env::var_os("RUST_LOG").is_none() {
            std::env::set_var("RUST_LOG", "example_multipart_form=debug,tower_http=debug")
        }
        tracing_subscriber::fmt::init();

        let clients = Clients {
            infer_cli: InferClient::connect("http://localhost:7999").await.unwrap(),
            process_cli: ProcessClient::connect("http://localhost:5001")
                .await
                .unwrap(),
        };

        let app = Router::new()
            .route("/", get(show_form).post(classify_image))
            .layer(AddExtensionLayer::new(clients))
            .layer(tower_http::trace::TraceLayer::new_for_http());

        let addr = "0.0.0.0:3000".parse().unwrap();
        tracing::info!("listening on {}", addr);
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
}

async fn show_form() -> Html<&'static str> {
    let html = std::include_str!("upload_form.html");

    Html(html)
}

async fn classify_image(
    ContentLengthLimit(mut multipart): ContentLengthLimit<
        Multipart,
        {
            250 * 1024 * 1024 /* 250mb */
        },
    >,
    Extension(Clients {
        mut infer_cli,
        mut process_cli,
    }): Extension<Clients>,
) -> impl IntoResponse {
    let mut results = vec![];
    while let Some(field) = multipart.next_field().await.unwrap() {
        let image = field.file_name().unwrap().to_string();
        let data = field.bytes().await.unwrap();

        // 调用预处理服务
        let PreProcessResponse { shape, data } = pre_process(&mut process_cli, &data).await;
        // 调用推理服务
        let InferResponse { shape, data } = infer(&mut infer_cli, shape, data).await;
        // 调用后处理服务
        let AfterProcessResponse { preds } = after_process(&mut process_cli, shape, data).await;
        let preds: Vec<_> = preds
            .into_iter()
            .map(|p| Pred {
                name: p.name,
                probability: p.probability,
            })
            .collect();
        results.push(Preds { image, preds })
    }
    Json(results)
}

async fn pre_process(cli: &mut ProcessClient<Channel>, data: &[u8]) -> PreProcessResponse {
    let req = PreProcessRequest { image: data.into() };
    cli.pre_process(req).await.unwrap().into_inner()
}

async fn infer(cli: &mut InferClient<Channel>, shape: Vec<u64>, data: Vec<f32>) -> InferResponse {
    let req = InferRequest { shape, data };
    cli.infer(req).await.unwrap().into_inner()
}

async fn after_process(
    cli: &mut ProcessClient<Channel>,
    shape: Vec<u64>,
    data: Vec<f32>,
) -> AfterProcessResponse {
    let req = AfterProcessRequest { shape, data };
    cli.after_process(req).await.unwrap().into_inner()
}

#[derive(Serialize, Deserialize)]
struct Pred {
    name: String,
    probability: f32,
}

#[derive(Serialize, Deserialize)]
struct Preds {
    image: String,
    preds: Vec<Pred>,
}
