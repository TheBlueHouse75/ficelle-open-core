from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request

from ficelle import router


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_serve_binds_before_catalog_warm(monkeypatch):
    """The HTTP server must bind and answer while the initial catalog warm is blocked.

    Regression for the startup hang (2026-06-26): ``serve()`` used to refresh the
    catalog *before* binding the server, so a single provider whose catalog endpoint
    hangs made the whole service unreachable at startup (``ficelle restart`` reported
    "did not become ready"). ``serve()`` now binds first and warms in the background.
    """
    port = _free_port()
    warm_started = threading.Event()
    release_warm = threading.Event()

    def _blocking_warm(config, force=False):
        # Simulate a provider catalog fetch that hangs well past any readiness window.
        warm_started.set()
        release_warm.wait(timeout=15)
        return {"models": [], "providers": {}}

    monkeypatch.setattr(router, "load_or_refresh_catalog", _blocking_warm)
    monkeypatch.setattr(router, "refresh_capability_oracle_if_stale", lambda config: None)
    monkeypatch.setattr(router, "auto_benchmark_loop", lambda config: None)

    servers: list = []
    orig_server_cls = router.ThreadingHTTPServer

    def _capture(*args, **kwargs):
        srv = orig_server_cls(*args, **kwargs)
        servers.append(srv)
        return srv

    monkeypatch.setattr(router, "ThreadingHTTPServer", _capture)

    config = router.load_config()
    config["host"] = "127.0.0.1"
    config["port"] = port

    thread = threading.Thread(target=router.serve, args=(config,), daemon=True)
    thread.start()
    try:
        # The server must answer within a few seconds — far below the warm's 15s block.
        deadline = time.time() + 5
        last_err: Exception | None = None
        responded = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/admin/status.json", timeout=1
                ):
                    responded = True
                    break
            except urllib.error.HTTPError:
                # The server answered (even with an error status) — it is serving, not hung.
                responded = True
                break
            except urllib.error.URLError as exc:  # not bound yet / connection refused
                last_err = exc
                time.sleep(0.1)
        assert responded, f"server did not serve while catalog warm was blocked: {last_err}"
        # Prove the warm really was still blocked when we got served (true bind-first).
        assert warm_started.is_set()
        assert not release_warm.is_set()
    finally:
        release_warm.set()
        for srv in servers:
            srv.shutdown()
            srv.server_close()


def test_catalog_refresh_loop_keeps_catalog_fresh(monkeypatch):
    """The catalog refresh daemon warms once, then force-refreshes well within the TTL,
    so an idle instance never lets the catalog go stale — which admin_status would
    otherwise report as a false '0 models / fail'. Regression for the long-idle-instance
    degradation that previously required a manual restart to clear."""
    from ficelle import router

    calls: list = []
    monkeypatch.setattr(router, "refresh_capability_oracle_if_stale", lambda config: calls.append(("oracle", None)))
    monkeypatch.setattr(router, "load_or_refresh_catalog", lambda config, force=False: calls.append(("refresh", force)))

    sleeps: list = []

    class _StopLoop(Exception):
        pass

    def _fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _StopLoop()

    monkeypatch.setattr(router.time, "sleep", _fake_sleep)

    try:
        router.catalog_refresh_loop({"catalog_ttl_seconds": 3600})
    except _StopLoop:
        pass

    refreshes = [call for call in calls if call[0] == "refresh"]
    # Warmed once on startup (no force), then forced at least one periodic refresh.
    assert refreshes[0] == ("refresh", False)
    assert ("refresh", True) in refreshes
    # Refreshes strictly more often than the TTL, so the catalog never crosses it.
    assert sleeps and all(interval < 3600 for interval in sleeps)


def test_catalog_refresh_interval_stays_below_ttl():
    """The refresh interval must stay strictly below the TTL for every TTL — otherwise
    the catalog goes stale between refreshes and the false '0 models / fail' returns. A
    malformed TTL must fall back to a sane default, never crash the refresh daemon."""
    from ficelle import router

    for ttl in (2, 30, 60, 120, 180, 600, 3600, 7200):
        interval = router.catalog_refresh_interval_seconds({"catalog_ttl_seconds": ttl})
        assert 0 < interval < ttl, f"interval {interval} must be in (0, {ttl})"

    # Malformed / missing TTL → safe default, never a crash that would kill the daemon.
    for bad in ("abc", None, "", "1h", {}):
        assert router.catalog_refresh_interval_seconds({"catalog_ttl_seconds": bad}) > 0
