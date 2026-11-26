use super::*;

pub(super) async fn private_asset(
    state: &AppState,
    parts: &[&str],
    media_type: &'static str,
) -> Result<Response> {
    let root = state.config.private_dir.as_ref().ok_or(NOT_FOUND)?;
    asset(
        root.clone(),
        parts.iter().map(|part| (*part).into()).collect(),
        media_type,
    )
    .await
}

pub(super) async fn index(State(state): State<AppState>) -> Result<Response> {
    let root = state.config.static_dir.ok_or(NOT_FOUND)?;
    asset(root, vec!["index.html".into()], "text/html; charset=utf-8").await
}

pub(super) async fn favicon(State(state): State<AppState>) -> Result<Response> {
    let root = state.config.static_dir.ok_or(NOT_FOUND)?;
    asset(root, vec!["favicon.ico".into()], "image/x-icon").await
}

pub(super) async fn public_asset(
    State(state): State<AppState>,
    Path(file): Path<String>,
) -> Result<Response> {
    let root = state.config.static_dir.ok_or(NOT_FOUND)?;
    let path = FsPath::new(&file);
    let parts: Vec<_> = file.split('/').collect();
    if parts
        .iter()
        .any(|part| part.is_empty() || part.starts_with('.') || part.contains(['\\', '%']))
        || parts
            .last()
            .is_some_and(|name| name.starts_with("house-plan") || name.starts_with("casa."))
    {
        return Err(NOT_FOUND);
    }
    let media_type = match path.extension().and_then(|extension| extension.to_str()) {
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("webp") => "image/webp",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("woff2") => "font/woff2",
        _ => return Err(NOT_FOUND),
    };
    let parts = std::iter::once("assets")
        .chain(parts)
        .map(str::to_owned)
        .collect();
    asset(root, parts, media_type).await
}

async fn asset(root: PathBuf, parts: Vec<String>, media_type: &'static str) -> Result<Response> {
    let file = tokio::task::spawn_blocking(move || {
        let (leaf, parents) = parts.split_last()?;
        let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut directory = open(&root, flags | OFlags::DIRECTORY, Mode::empty()).ok()?;
        for name in parents {
            directory = openat(&directory, name, flags | OFlags::DIRECTORY, Mode::empty()).ok()?;
        }
        let file =
            File::from(openat(directory, leaf, flags | OFlags::NONBLOCK, Mode::empty()).ok()?);
        let metadata = file.metadata().ok()?;
        metadata.is_file().then_some((file, metadata.len()))
    })
    .await
    .map_err(|_| NOT_FOUND)?
    .ok_or(NOT_FOUND)?;
    let headers = [
        (header::CONTENT_TYPE, HeaderValue::from_static(media_type)),
        (header::CONTENT_LENGTH, HeaderValue::from(file.1)),
        (
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ),
    ];
    let stream = ReaderStream::new(tokio::fs::File::from_std(file.0));
    Ok((headers, Body::from_stream(stream)).into_response())
}
