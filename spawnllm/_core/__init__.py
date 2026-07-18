"""Thin host binding to the embedded spawnllm-core wasm blob.

The Rust core owns all drift-prone logic behind `dispatch(op_json) -> result_json`;
this module loads the wasm32-wasip1 blob under wasmtime and marshals JSON in and out.
wasmtime-py exposes no in-memory WASI stderr sink, so a panicking core's diagnostics
are routed to a process-lifetime temp file whose tail is read back into the raised
`CoreTrap`. The core builds with panic=abort, so a trap leaves the allocator in an
untrustworthy state: the cached runtime is dropped and the next call re-instantiates.
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wasmtime import Engine, Func, Linker, Memory, Module, Store, Trap, WasiConfig, WasmtimeError

if TYPE_CHECKING:
    from wasmtime._instance import InstanceExports

WASM_NAME = "spawnllm_core.wasm"
STDERR_TAIL_BYTES = 4096
MISSING_BLOB = (
    f"spawnllm core wasm blob missing: {WASM_NAME} was not packaged. Build it with "
    "`bash scripts/build_wasm.sh` (run `rustup target add wasm32-wasip1` first)."
)

__all__ = ["CoreError", "CoreTrap", "dispatch", "dispatch_raw", "version"]

LOCK = threading.Lock()
RUNTIME: Runtime | None = None


class CoreError(Exception):
    def __init__(self, kind: str, msg: str) -> None:
        super().__init__(f"{kind}: {msg}")
        self.kind = kind
        self.msg = msg


class CoreTrap(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Runtime:
    store: Store
    alloc: Func
    free: Func
    call: Func
    memory: Memory
    stderr_path: Path

    def invoke(self, request_json: str) -> str:
        request = request_json.encode()
        request_len = len(request)
        request_ptr = self.alloc(self.store, request_len)
        try:
            self.memory.write(self.store, request, request_ptr)
            packed = self.call(self.store, request_ptr, request_len)
        except (Trap, WasmtimeError):
            raise
        except BaseException:
            self.free(self.store, request_ptr, request_len)
            raise
        self.free(self.store, request_ptr, request_len)
        response_ptr, response_len = packed >> 32, packed & 0xFFFFFFFF
        try:
            response = bytes(self.memory.read(self.store, response_ptr, response_ptr + response_len))
        except (Trap, WasmtimeError):
            raise
        except BaseException:
            self.free(self.store, response_ptr, response_len)
            raise
        self.free(self.store, response_ptr, response_len)
        return response.decode()


def export[T](exports: InstanceExports, name: str, kind: type[T]) -> T:
    value = exports[name]
    assert isinstance(value, kind), f"core export {name} is {type(value).__name__}, not {kind.__name__}"
    return value


def build() -> Runtime:
    blob = files(__package__).joinpath(WASM_NAME)
    if not blob.is_file():
        raise FileNotFoundError(MISSING_BLOB)
    engine = Engine()
    linker = Linker(engine)
    linker.define_wasi()
    store = Store(engine)
    stderr = Path((fd_name := tempfile.mkstemp(prefix="spawnllm-core-", suffix=".stderr"))[1])
    os.close(fd_name[0])
    wasi = WasiConfig()
    wasi.stderr_file = str(stderr)
    store.set_wasi(wasi)
    exports = linker.instantiate(store, Module(engine, blob.read_bytes())).exports(store)
    if (initialize := exports.get("_initialize")) is not None:
        initialize(store)
    return Runtime(
        store=store,
        alloc=export(exports, "sl_alloc", Func),
        free=export(exports, "sl_free", Func),
        call=export(exports, "sl_call", Func),
        memory=export(exports, "memory", Memory),
        stderr_path=stderr,
    )


def load() -> Runtime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = build()
    return RUNTIME


def reset() -> None:
    global RUNTIME
    if RUNTIME is not None:
        RUNTIME.stderr_path.unlink(missing_ok=True)
    RUNTIME = None


def fork_reset() -> None:
    global LOCK, RUNTIME
    LOCK = threading.Lock()
    RUNTIME = None


atexit.register(reset)
os.register_at_fork(after_in_child=fork_reset)


def trap_message(runtime: Runtime, error: Exception) -> str:
    tail = runtime.stderr_path.read_bytes()[-STDERR_TAIL_BYTES:].decode(errors="replace").strip()
    return f"spawnllm core trap: {error}\nwasm stderr: {tail}" if tail else f"spawnllm core trap: {error}"


def dispatch_raw(request_json: str) -> str:
    with LOCK:
        runtime = load()
        try:
            return runtime.invoke(request_json)
        except (Trap, WasmtimeError) as error:
            message = trap_message(runtime, error)
            reset()
            raise CoreTrap(message) from error


def dispatch(op: str, input: dict | None = None) -> Any:
    envelope = json.loads(dispatch_raw(json.dumps({"op": op} | ({"input": input} if input is not None else {}))))
    if (error := envelope.get("err")) is not None:
        raise CoreError(error["kind"], error["msg"])
    return envelope["ok"]


def version() -> dict:
    return dispatch("version")
