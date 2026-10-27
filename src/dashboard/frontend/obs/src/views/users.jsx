// React owns only the container. Voice names, transcripts and audio stay in
// the scoped imperative view and its explicit retirement owners.
function UsersView() {
  const container = React.useRef(null);
  React.useLayoutEffect(() => window.KzaVoiceView.mountVoices(
    container.current,window.__kzaVoiceController),[]);
  return <div className="view" id="view-users" ref={container} />;
}
window.UsersView = UsersView;
