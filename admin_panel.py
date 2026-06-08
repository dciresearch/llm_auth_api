import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api_database import Database, UserAuth
from src.utils import load_global_config

logger = logging.getLogger(__name__)

CFG = load_global_config()
ADMIN_SECRET = CFG.get('admin_config', {}).get('admin_secret', '')
ADMIN_PORT = CFG.get('admin_config', {}).get('admin_port', 6334)
db_path = "./database/generic.db"
api_db = Database(db_path)

app = FastAPI(title="LLM Auth API — Admin Panel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(request: Request):
    if not ADMIN_SECRET:
        return True
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    cookie_token = request.cookies.get("admin_token", "")
    if token == ADMIN_SECRET or cookie_token == ADMIN_SECRET:
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")


def _ts_to_str(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _str_to_ts(s):
    if not s or s.strip() == "":
        return None
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return int(s) if s.isdigit() else None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Auth API — Admin Panel</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
          --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149;
          --yellow: #d29922; --hover: #1f2937; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 1.5rem; margin-bottom: 20px; color: var(--accent); }
  h2 { font-size: 1.2rem; margin: 20px 0 10px; color: var(--muted); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 16px; margin-bottom: 16px; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: var(--bg); color: var(--muted); font-weight: 600; text-align: left;
       padding: 8px 10px; border-bottom: 2px solid var(--border); white-space: nowrap; }
  td { padding: 6px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:hover td { background: var(--hover); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px;
           font-weight: 600; }
  .badge-active { background: #1a4d2e; color: var(--green); }
  .badge-inactive { background: #4d1a1a; color: var(--red); }
  .badge-expired { background: #4d3a1a; color: var(--yellow); }
  .badge-unlimited { background: #1a2d4d; color: var(--accent); }
  input, select { background: var(--bg); color: var(--text); border: 1px solid var(--border);
                  border-radius: 4px; padding: 4px 8px; font-size: 13px; width: 100%; }
  input[type="number"] { width: 100px; }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  button { background: var(--accent); color: #fff; border: none; border-radius: 4px;
           padding: 6px 14px; font-size: 13px; cursor: pointer; font-weight: 600; }
  button:hover { opacity: 0.9; }
  button.btn-danger { background: var(--red); }
  button.btn-green { background: var(--green); }
  button.btn-small { padding: 3px 10px; font-size: 12px; }
  .actions { display: flex; gap: 4px; }
  .key-cell { max-width: 200px; overflow: hidden; text-overflow: ellipsis;
              white-space: nowrap; font-family: monospace; font-size: 11px; }
  .stats { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 12px 20px; min-width: 140px; }
  .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
  .stat-label { font-size: 12px; color: var(--muted); }
  .flash { position: fixed; top: 10px; right: 10px; padding: 10px 20px; border-radius: 6px;
           font-size: 13px; z-index: 999; animation: fadeOut 3s forwards; }
  .flash-ok { background: #1a4d2e; color: var(--green); border: 1px solid var(--green); }
  .flash-err { background: #4d1a1a; color: var(--red); border: 1px solid var(--red); }
  @keyframes fadeOut { 0%,70% { opacity:1; } 100% { opacity:0; } }
  .new-row td { background: var(--hover); }
  .new-row input { background: var(--card); }
</style>
</head>
<body>
<h1>LLM Auth API — Admin Panel</h1>
<div class="stats">
  <div class="stat"><div class="stat-value">{{ total_users }}</div><div class="stat-label">Total tokens</div></div>
  <div class="stat"><div class="stat-value">{{ active_users }}</div><div class="stat-label">Active</div></div>
  <div class="stat"><div class="stat-value">{{ inactive_users }}</div><div class="stat-label">Inactive</div></div>
  <div class="stat"><div class="stat-value">{{ total_requests }}</div><div class="stat-label">Total requests</div></div>
</div>

<h2>Tokens</h2>
<div class="card">
<table>
<thead>
<tr>
  <th>ID</th>
  <th>Name</th>
  <th>Key</th>
  <th>Priority</th>
  <th>Active</th>
  <th>Created</th>
  <th>Expires</th>
  <th>Allowed Models</th>
  <th>Tokens Used</th>
  <th>Budget</th>
  <th>Lim/min</th>
  <th>Lim/hour</th>
  <th>Lim/day</th>
  <th>Req/min</th>
  <th>Actions</th>
</tr>
</thead>
<tbody>
{% for u in users %}
<tr data-id="{{ u.id }}">
  <td>{{ u.id }}</td>
  <td><input type="text" value="{{ u.user_name }}" data-field="user_name" style="width:120px"></td>
  <td class="key-cell"><span title="{{ u.user_key }}">{{ u.user_key[:16] }}...</span> <button class="btn-small" onclick="copyKey('{{ u.user_key }}', this)" title="Copy full key">Copy</button></td>
  <td><input type="number" value="{{ u.priority }}" data-field="priority" style="width:60px" min="0" max="10"></td>
  <td>
    {% if u.is_active %}
      <span class="badge badge-active">active</span>
    {% else %}
      <span class="badge badge-inactive">inactive</span>
    {% endif %}
  </td>
  <td style="font-size:11px;white-space:nowrap">{{ u.created_at_str }}</td>
  <td><input type="datetime-local" value="{{ u.expires_at_input }}" data-field="expires_at" style="width:170px"></td>
  <td><input type="text" value="{{ u.allowed_models or '' }}" data-field="allowed_models" placeholder="all" style="width:130px"></td>
  <td style="font-weight:600">{{ "{:,}".format(u.total_tokens_used) }}</td>
  <td><input type="number" value="{{ u.token_budget or '' }}" data-field="token_budget" placeholder="unlimited" style="width:100px" min="0"></td>
  <td><input type="number" value="{{ u.rate_limit_tokens_per_min or '' }}" data-field="rate_limit_tokens_per_min" placeholder="-" style="width:80px" min="0"></td>
  <td><input type="number" value="{{ u.rate_limit_tokens_per_hour or '' }}" data-field="rate_limit_tokens_per_hour" placeholder="-" style="width:80px" min="0"></td>
  <td><input type="number" value="{{ u.rate_limit_tokens_per_day or '' }}" data-field="rate_limit_tokens_per_day" placeholder="-" style="width:80px" min="0"></td>
  <td><input type="number" value="{{ u.rate_limit_requests_per_min or '' }}" data-field="rate_limit_requests_per_min" placeholder="-" style="width:80px" min="0"></td>
  <td>
    <div class="actions">
      <button class="btn-small btn-green" onclick="saveRow({{ u.id }}, this)">Save</button>
      <button class="btn-small" onclick="resetUsage({{ u.id }})">Reset usage</button>
      {% if u.is_active %}
        <button class="btn-small btn-danger" onclick="toggleActive({{ u.id }}, false)">Revoke</button>
      {% else %}
        <button class="btn-small btn-green" onclick="toggleActive({{ u.id }}, true)">Activate</button>
      {% endif %}
    </div>
  </td>
</tr>
{% endfor %}
<tr class="new-row">
  <td>NEW</td>
  <td><input type="text" id="new-name" placeholder="user name" style="width:120px"></td>
  <td><input type="text" id="new-key" placeholder="auto-generated" style="width:130px;font-size:11px"></td>
  <td><input type="number" id="new-priority" value="5" style="width:60px" min="0" max="10"></td>
  <td colspan="11">
    <button class="btn-green" onclick="createToken()">+ Create Token</button>
  </td>
</tr>
</tbody>
</table>
</div>

<div id="flash"></div>

<script>
function flash(msg, ok) {
  const el = document.getElementById('flash');
  el.innerHTML = `<div class="flash ${ok ? 'flash-ok' : 'flash-err'}">${msg}</div>`;
  setTimeout(() => el.innerHTML = '', 3500);
}

function copyKey(key, btn) {
  navigator.clipboard.writeText(key).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = orig, 1500);
  }).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = key;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

function getRowData(id) {
  const row = document.querySelector(`tr[data-id="${id}"]`);
  const data = {};
  row.querySelectorAll('input[data-field]').forEach(inp => {
    let v = inp.value.trim();
    if (inp.type === 'number' && v === '') v = null;
    else if (inp.type === 'number') v = parseInt(v);
    else if (inp.type === 'datetime-local' && v === '') v = null;
    data[inp.dataset.field] = v;
  });
  return data;
}

async function saveRow(id, btn) {
  const data = getRowData(id);
  btn.disabled = true;
  btn.textContent = '...';
  try {
    const r = await fetch(`/api/users/${id}`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const j = await r.json();
    if (r.ok) flash('Saved', true);
    else flash(j.detail || 'Error', false);
  } catch(e) { flash(e.message, false); }
  btn.disabled = false;
  btn.textContent = 'Save';
}

async function resetUsage(id) {
  if (!confirm('Reset token usage counter to 0?')) return;
  const r = await fetch(`/api/users/${id}/reset-usage`, {method: 'POST'});
  if (r.ok) { flash('Usage reset', true); location.reload(); }
  else flash('Error', false);
}

async function toggleActive(id, active) {
  const r = await fetch(`/api/users/${id}`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({is_active: active ? 1 : 0})
  });
  if (r.ok) { flash(active ? 'Activated' : 'Revoked', true); location.reload(); }
  else flash('Error', false);
}

async function createToken() {
  const name = document.getElementById('new-name').value.trim();
  const key = document.getElementById('new-key').value.trim() || null;
  const priority = parseInt(document.getElementById('new-priority').value) || 5;
  if (!name) { flash('Name is required', false); return; }
  const r = await fetch('/api/users', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_name: name, priority, key})
  });
  if (r.ok) { flash('Token created', true); location.reload(); }
  else { const j = await r.json(); flash(j.detail || 'Error', false); }
}
</script>
</body>
</html>"""


def _user_to_dict(u):
    return {
        "id": u.id,
        "user_name": u.user_name,
        "user_key": u.user_key,
        "priority": u.priority,
        "allowed_models": u.allowed_models,
        "created_at": u.created_at,
        "created_at_str": _ts_to_str(u.created_at),
        "expires_at": u.expires_at,
        "expires_at_input": datetime.fromtimestamp(u.expires_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M") if u.expires_at else "",
        "is_active": u.is_active,
        "total_tokens_used": u.total_tokens_used or 0,
        "token_budget": u.token_budget,
        "rate_limit_tokens_per_min": u.rate_limit_tokens_per_min,
        "rate_limit_tokens_per_hour": u.rate_limit_tokens_per_hour,
        "rate_limit_tokens_per_day": u.rate_limit_tokens_per_day,
        "rate_limit_requests_per_min": u.rate_limit_requests_per_min,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, _=Depends(_check_auth)):
    from sqlalchemy import func as sa_func
    from src.api_database import Requests

    with api_db.Session() as session:
        users = session.query(UserAuth).order_by(UserAuth.id).all()
        user_dicts = [_user_to_dict(u) for u in users]
        total_users = len(user_dicts)
        active_users = sum(1 for u in user_dicts if u["is_active"])
        total_requests = session.query(sa_func.count(Requests.id)).scalar() or 0

    from jinja2 import Template
    tmpl = Template(HTML_TEMPLATE)
    return tmpl.render(
        users=user_dicts,
        total_users=total_users,
        active_users=active_users,
        inactive_users=total_users - active_users,
        total_requests=total_requests,
    )


@app.get("/api/users")
async def api_list_users(_=Depends(_check_auth)):
    with api_db.Session() as session:
        users = session.query(UserAuth).order_by(UserAuth.id).all()
        return [_user_to_dict(u) for u in users]


@app.post("/api/users")
async def api_create_user(request: Request, _=Depends(_check_auth)):
    data = await request.json()
    name = data.get("user_name")
    priority = data.get("priority", 5)
    key = data.get("key")
    if not name:
        raise HTTPException(400, "user_name is required")
    user_name, priority, new_key = api_db.register_new_user(name, priority, key)
    return {"user_name": user_name, "priority": priority, "key": new_key}


@app.patch("/api/users/{user_id}")
async def api_update_user(user_id: int, request: Request, _=Depends(_check_auth)):
    data = await request.json()
    with api_db.Session() as session:
        user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
        if not user:
            raise HTTPException(404, "User not found")

        if "user_name" in data:
            user.user_name = data["user_name"]
        if "priority" in data:
            user.priority = data["priority"]
        if "is_active" in data:
            user.is_active = data["is_active"]
        if "allowed_models" in data:
            v = data["allowed_models"]
            user.allowed_models = None if v in (None, "", "all", "null") else v
        if "expires_at" in data:
            v = data["expires_at"]
            if v is None or v == "":
                user.expires_at = None
            elif isinstance(v, str):
                user.expires_at = _str_to_ts(v)
            elif isinstance(v, (int, float)):
                user.expires_at = int(v)
        if "token_budget" in data:
            v = data["token_budget"]
            user.token_budget = None if v in (None, "", 0) else int(v)
        if "rate_limit_tokens_per_min" in data:
            v = data["rate_limit_tokens_per_min"]
            user.rate_limit_tokens_per_min = None if v in (None, "", 0) else int(v)
        if "rate_limit_tokens_per_hour" in data:
            v = data["rate_limit_tokens_per_hour"]
            user.rate_limit_tokens_per_hour = None if v in (None, "", 0) else int(v)
        if "rate_limit_tokens_per_day" in data:
            v = data["rate_limit_tokens_per_day"]
            user.rate_limit_tokens_per_day = None if v in (None, "", 0) else int(v)
        if "rate_limit_requests_per_min" in data:
            v = data["rate_limit_requests_per_min"]
            user.rate_limit_requests_per_min = None if v in (None, "", 0) else int(v)

        session.commit()
        return _user_to_dict(user)


@app.post("/api/users/{user_id}/reset-usage")
async def api_reset_usage(user_id: int, _=Depends(_check_auth)):
    api_db.reset_token_usage(user_id)
    return {"status": "ok"}
