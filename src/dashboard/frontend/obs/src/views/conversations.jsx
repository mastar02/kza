// React owns only the container; private rows are never props or state.
function ConversationsView() {
  const container = React.useRef(null);
  React.useLayoutEffect(() => window.KzaConversationPrivacy.mountConversations(
    container.current, window.__kzaConversationState), []);
  return <div className="view" id="view-conversations" ref={container} />;
}
window.ConversationsView = ConversationsView;
