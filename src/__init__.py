# Home Assistant Voice Control
# G1 (U3b-0e): an unmanaged entry under a managed release root is refused here,
# at the first import of the package, before any module can open a store.
from src.core.entry_guard import guard_package_entry

guard_package_entry()
