import docker

client = docker.from_env()
prefix = "dockermanaged_vllm"

containers = client.containers.list(all=True)
removed = 0
for c in containers:
    if c.name.startswith(prefix) and c.status in ('exited', 'dead'):
        print(f"Removing: {c.name} (status={c.status})")
        try:
            c.remove(force=True)
            removed += 1
        except Exception as e:
            print(f"  Failed: {e}")

print(f"\nRemoved {removed} dead container(s)")
