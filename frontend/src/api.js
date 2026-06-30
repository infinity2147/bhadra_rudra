// In dev, vite proxies /api → http://localhost:8000 (see vite.config.js)
const API = '';

// Current role is held in localStorage so it survives page reloads.
// The Role context (see App.jsx) reads/writes this value.
export function getRole() {
  return localStorage.getItem('rudra_role') || 'INVESTIGATOR';
}

export function setRole(role) {
  localStorage.setItem('rudra_role', role);
}

function withRoleHeaders(extra = {}) {
  return { 'X-User-Role': getRole(), ...extra };
}

export async function fetchAPI(path) {
  const res = await fetch(`${API}${path}`, { headers: withRoleHeaders() });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API error: ${res.status} ${text}`);
  }
  return res.json();
}

export async function postAPI(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: withRoleHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    const err = new Error(`API error: ${res.status} ${text}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export function apiUrl(path) {
  return `${API}${path}`;
}

export const getIncidentRca = (id) => fetchAPI(`/api/incidents/${id}/rca`);

export async function downloadFromAPI(path, filename) {
  const res = await fetch(`${API}${path}`, { headers: withRoleHeaders() });
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
