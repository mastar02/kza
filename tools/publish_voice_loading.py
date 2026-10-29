"""Publish the verified voice loading UI release through the existing release flow."""

from __future__ import annotations

try:
    import publish_voice_order as release
except ModuleNotFoundError:
    from tools import publish_voice_order as release


def configure_release() -> None:
    """Pin this operation to the tested artifact and its exact two frontend changes."""
    release.COMMIT = 'e9669ec1edb101ca6b190ef8df09314e1ac63562'
    release.PREVIOUS_COMMIT = '63ea8d319b50506cc37ba73df1e41c87a41054cf'
    release.ARCHIVE_SHA256 = 'e4fd2fb871c11fd17671b0e68329c2ccaf46dbf6ea289057ca6a57b38b2d08e5'
    release.RELEASE_TAG = 'voice-loading'
    release.APP_CHANGES = {'src/dashboard/frontend/obs/src/voice-view.js',
                           'src/dashboard/frontend/obs/src/voice-channel.js'}


if __name__ == '__main__':
    configure_release()
    release.main()
