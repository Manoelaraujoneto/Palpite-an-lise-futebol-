"""
Palpites: Over 2.5 gols + Ambas marcam (BTTS), com busca por data.

Roda no Pydroid: abra e toque em play. Baixa os dados sozinho (precisa de internet).
Gera a página:  /storage/emulated/0/dados_palpites.html  (abra no Chrome)

Requer: pandas, scikit-learn
"""
import os, sys, glob, json, warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

PASTA = "/storage/emulated/0/dados"
BASE = "https://www.football-data.co.uk"
LIGAS = ["E0", "E1", "SC0", "D1", "I1", "SP1", "F1", "N1", "P1", "B1", "T1", "G1"]
TEMPORADAS = ["2122", "2223", "2324", "2425", "2526", "2627"]
N, MIN_GAMES = 10, 3

NOMES = {
    "E0": "Premier League (ING)", "E1": "Championship (ING)", "E2": "League One (ING)",
    "E3": "League Two (ING)", "EC": "Conference (ING)", "SC0": "Premiership (ESC)",
    "SC1": "Championship (ESC)", "D1": "Bundesliga (ALE)", "D2": "2. Bundesliga (ALE)",
    "I1": "Serie A (ITA)", "I2": "Serie B (ITA)", "SP1": "La Liga (ESP)",
    "SP2": "La Liga 2 (ESP)", "F1": "Ligue 1 (FRA)", "F2": "Ligue 2 (FRA)",
    "N1": "Eredivisie (HOL)", "P1": "Primeira Liga (POR)", "B1": "Pro League (BEL)",
    "T1": "Süper Lig (TUR)", "G1": "Super League (GRE)",
}


# ---------- dados ----------
def baixar(url, destino):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        dados = urllib.request.urlopen(req, timeout=30).read()
        if len(dados) < 500:
            return False
        with open(destino, "wb") as f:
            f.write(dados)
        return True
    except Exception:
        return False


def garantir_dados(pasta):
    os.makedirs(pasta, exist_ok=True)
    if glob.glob(f"{pasta}/*.csv"):
        return
    print("Baixando históricos (1ª vez, 1-2 min)...")
    ok = sum(baixar(f"{BASE}/mmz4281/{t}/{l}.csv", f"{pasta}/{l}_{t}.csv")
             for t in TEMPORADAS for l in LIGAS)
    print(f"{ok} arquivos baixados")


def limpar_colunas(df):
    df.columns = [str(c).encode("ascii", "ignore").decode().strip() for c in df.columns]
    return df


def carregar(pasta):
    arqs = glob.glob(f"{pasta}/*.csv") + glob.glob(f"{pasta}/*.CSV")
    if not arqs:
        sys.exit(f"Nenhum CSV em {pasta}. Verifique a internet e rode de novo.")
    dfs = [limpar_colunas(pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")) for f in arqs]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["FTHG"] = pd.to_numeric(df.FTHG, errors="coerce")
    df["FTAG"] = pd.to_numeric(df.FTAG, errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"]).reset_index(drop=True)
    df["y_over"] = ((df.FTHG + df.FTAG) > 2.5).astype(int)
    df["y_btts"] = ((df.FTHG > 0) & (df.FTAG > 0)).astype(int)
    return df


# ---------- features (só jogos anteriores) ----------
class Features:
    def __init__(self):
        self.team = defaultdict(lambda: deque(maxlen=N))
        self.league = defaultdict(lambda: deque(maxlen=300))

    def _t(self, div, team):
        d = self.team[(div, team)]
        if len(d) < MIN_GAMES:
            return [np.nan] * 7
        a = np.array(d, dtype=float)
        gf, ga = a[:, 0], a[:, 1]
        return [gf.mean(), ga.mean(), ((gf + ga) > 2.5).mean(), (gf + ga).mean(),
                ((gf > 0) & (ga > 0)).mean(), (gf > 0).mean(), (ga == 0).mean()]

    def row(self, div, home, away):
        h, a = self._t(div, home), self._t(div, away)
        lg = self.league[div]
        if len(lg) >= 20:
            m = np.array(lg, dtype=float).mean(axis=0)
        else:
            m = [np.nan] * 3
        eh, ea = (h[0] + a[1]) / 2, (a[0] + h[1]) / 2
        return h + a + list(m) + [eh + ea, eh, ea]

    def update(self, div, home, away, hg, ag):
        self.team[(div, home)].append((hg, ag))
        self.team[(div, away)].append((ag, hg))
        self.league[div].append((hg + ag, int(hg + ag > 2.5), int(hg > 0 and ag > 0)))


def montar(df):
    fs, X = Features(), []
    for r in df.itertuples():
        X.append(fs.row(r.Div, r.HomeTeam, r.AwayTeam))
        fs.update(r.Div, r.HomeTeam, r.AwayTeam, r.FTHG, r.FTAG)
    return np.array(X, dtype=float), fs


def modelo():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                          max_iter=250, l2_regularization=1.0)


def confiabilidade(y, p):
    """Nos jogos de teste: quando o modelo deu >= X%, quanto acertou de fato."""
    res = {}
    for thr in (60, 70, 80, 90):
        m = p >= thr / 100
        res[str(thr)] = [int(m.sum()), float(y[m].mean()) if m.sum() else None]
    return res


# ---------- página ----------
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Palpites - Over 2.5 e Ambas marcam</title>
<style>
:root{--bg:#f4f6f8;--card:#fff;--tx:#1b1f24;--mut:#6b7480;--bd:#e3e7ec;--ok:#1a9e55;--mid:#e0a100;--low:#b8c0ca;--ac:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#12151a;--card:#1b2027;--tx:#e9edf2;--mut:#8f9aa8;--bd:#2a313a;--low:#4a5360;--ac:#5b8cff}}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--tx)}
header{padding:14px 12px 2px}h1{margin:0;font-size:19px}.sub{color:var(--mut);font-size:12px;margin-top:2px}
.abas{display:flex;gap:6px;padding:10px 12px 0}
.aba{flex:1;padding:11px 6px;border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:10px;font-size:14px;font-weight:600}
.aba.on{background:var(--ac);color:#fff;border-color:var(--ac)}
.filtros{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 12px}
select{width:100%;background:var(--card);color:var(--tx);border:1px solid var(--bd);border-radius:8px;padding:10px 8px;font-size:13px}
.conf{margin:0 12px 8px;padding:9px 11px;background:var(--card);border:1px solid var(--bd);border-radius:10px;font-size:12px;color:var(--mut);line-height:1.6}
.conf b{color:var(--tx)}
.wrap{padding:0 8px 8px}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:10px;overflow:hidden}
th{font-size:10px;text-transform:uppercase;color:var(--mut);text-align:left;padding:8px 5px;border-bottom:1px solid var(--bd);white-space:nowrap}
td{padding:9px 5px;font-size:13px;border-bottom:1px solid var(--bd);vertical-align:middle}
tr:last-child td{border:none}
.jogo{font-weight:600;line-height:1.25}.liga{color:var(--mut);font-size:11px;margin-top:2px}
.bar{height:6px;background:var(--bd);border-radius:4px;margin-top:4px;min-width:44px}
.bar i{display:block;height:100%;border-radius:4px}
.pct{font-weight:700;font-size:14px}.val{font-weight:700;white-space:nowrap}.pos{color:var(--ok)}.neg{color:var(--mut);font-weight:400}
.vazio{padding:18px 12px;text-align:center;color:var(--mut);line-height:1.5}
.leg{padding:6px 12px 24px;color:var(--mut);font-size:12px;line-height:1.5}.leg b{color:var(--tx)}
</style></head><body>
<header><a id="upd" href="/atualizar" onclick="this.textContent='⏳ Atualizando... aguarde'" style="display:none;float:right;padding:8px 10px;border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:8px;text-decoration:none;font-size:12px">🔄 Atualizar</a><h1>⚽ Palpites</h1><div class="sub">Probabilidades estimadas pelo modelo · atualizado __ATUALIZADO__</div></header>
<div class="abas">
<button class="aba on" id="a-over">Over 2.5 gols</button>
<button class="aba" id="a-btts">Ambas marcam</button>
</div>
<div class="filtros">
<select id="data"></select>
<select id="liga"><option value="">Todas as ligas</option></select>
<select id="min">
<option value="0">Qualquer chance</option><option value="0.6">Chance 60%+</option>
<option value="0.7">Chance 70%+</option><option value="0.8">Chance 80%+</option><option value="0.9">Chance 90%+</option>
</select>
<select id="odd">
<option value="">Qualquer odd da casa</option><option value="0-1.5">Odd até 1.50</option>
<option value="1.5-1.8">Odd 1.50 a 1.80</option><option value="1.8-2.2">Odd 1.80 a 2.20</option><option value="2.2-99">Odd 2.20 ou mais</option>
</select>
<select id="val"><option value="">Com ou sem valor</option><option value="1">Só com valor</option></select>
</div>
<div class="conf" id="conf"></div>
<div class="wrap"><table><thead id="th"></thead><tbody id="tb"></tbody></table><div class="vazio" id="vz" style="display:none"></div></div>
<div class="leg"><b>Chance:</b> probabilidade estimada de o palpite acontecer.
<b>Justa:</b> odd que o palpite deveria pagar com essa chance (1 ÷ chance).
<b>Casa:</b> odd da casa de apostas (só Over 2.5).
<b>Valor:</b> verde quando a casa paga mais que a odd justa.
<b>Teste do modelo:</b> mostra o quanto o modelo realmente acertou em jogos passados quando deu aquela chance. Nenhuma chance é garantia de resultado.</div>
<script>
const D=__DADOS__,S=__STATS__;
const DS=['Dom','Seg','Ter','Qua','Qui','Sex','Sáb'];
let aba='over';
if(['localhost','127.0.0.1'].indexOf(location.hostname)>=0)document.getElementById('upd').style.display='';
const $=id=>document.getElementById(id);
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function lbl(iso){const d=new Date(iso+'T12:00:00');return DS[d.getDay()]+' '+iso.slice(8,10)+'/'+iso.slice(5,7)}
const sd=$('data');sd.innerHTML='<option value="">Todas as datas</option>';
[...new Set(D.map(x=>x.d))].sort().forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=lbl(d);sd.appendChild(o)});
[...new Set(D.map(x=>x.liga))].sort().forEach(l=>{const o=document.createElement('option');o.value=l;o.textContent=l;$('liga').appendChild(o)});
const cor=p=>p>=0.7?'var(--ok)':p>=0.55?'var(--mid)':'var(--low)';
function conf(){
 const s=S[aba],t=[];
 for(const k of ['60','70','80','90']){const [n,h]=s[k];t.push('Chance '+k+'%+: '+(n?'acertou <b>'+Math.round(h*100)+'%</b> ('+n+' jogos)':'<b>nenhum jogo</b> chegou a esse nível'))}
 $('conf').innerHTML='<b>Teste do modelo em jogos passados</b><br>'+t.join('<br>');
}
function draw(){
 const P=aba==='over'?'po':'pb',J=aba==='over'?'jo':'jb',dt=sd.value,L=$('liga').value,M=+$('min').value,V=$('val').value==='1'&&aba==='over',O=$('odd').value;
 $('odd').style.display=$('val').style.display=aba==='over'?'':'none';
 $('th').innerHTML='<tr><th>Jogo</th><th>Chance</th><th>Justa</th>'+(aba==='over'?'<th>Casa</th><th>Valor</th>':'')+'</tr>';
 const rows=D.filter(x=>(!dt||x.d===dt)&&(!L||x.liga===L)&&x[P]>=M&&(!V||(x.edge!==null&&x.edge>0))&&(aba!=='over'||!O||(x.odd!==null&&x.odd>=+O.split('-')[0]&&x.odd<+O.split('-')[1]))).sort((a,b)=>b[P]-a[P]);
 $('tb').innerHTML=rows.map(x=>{
  let extra='';
  if(aba==='over'){const ed=x.edge===null?'-':'<span class="val '+(x.edge>0?'pos':'neg')+'">'+(x.edge>0?'+':'')+(x.edge*100).toFixed(1)+'%</span>';
   extra='<td>'+(x.odd===null?'-':x.odd.toFixed(2))+'</td><td>'+ed+'</td>'}
  return '<tr><td><div class="jogo">'+esc(x.casa)+' x '+esc(x.fora)+'</div><div class="liga">'+lbl(x.d)+(x.h?' '+esc(x.h):'')+' · '+esc(x.liga)+'</div></td>'+
  '<td><div class="pct">'+Math.round(x[P]*100)+'%</div><div class="bar"><i style="width:'+Math.round(x[P]*100)+'%;background:'+cor(x[P])+'"></i></div></td>'+
  '<td>'+x[J].toFixed(2)+'</td>'+extra+'</tr>'}).join('');
 const vz=$('vz');
 if(!rows.length){vz.style.display='';vz.innerHTML=M>=0.9?'Nenhum jogo com chance de 90%+ nesse filtro.<br>O modelo quase nunca chega a esse nível porque futebol tem muito acaso. Tente 70%+ ou 80%+.':'Nenhum jogo com esse filtro.'}else vz.style.display='none';
 conf();
}
function troca(a){aba=a;$('a-over').classList.toggle('on',a==='over');$('a-btts').classList.toggle('on',a==='btts');draw()}
$('a-over').onclick=()=>troca('over');$('a-btts').onclick=()=>troca('btts');
['data','liga','min','odd','val'].forEach(i=>$(i).onchange=draw);draw();
</script></body></html>"""


def gerar_html(fx, stats, caminho):
    dados = []
    for r in fx.itertuples():
        odd = getattr(r, "OddCasa", np.nan)
        edge = getattr(r, "Edge", np.nan)
        hora = getattr(r, "Time", None)
        dados.append({
            "d": r.Data.strftime("%Y-%m-%d"), "h": hora if isinstance(hora, str) else "",
            "liga": NOMES.get(r.Div, r.Div), "casa": str(r.HomeTeam), "fora": str(r.AwayTeam),
            "po": float(r.P_over), "pb": float(r.P_btts),
            "jo": float(1 / max(r.P_over, 0.01)), "jb": float(1 / max(r.P_btts, 0.01)),
            "odd": None if pd.isna(odd) else float(odd),
            "edge": None if pd.isna(edge) else float(edge),
        })
    html = (HTML.replace("__DADOS__", json.dumps(dados, ensure_ascii=False))
                .replace("__STATS__", json.dumps(stats))
                .replace("__ATUALIZADO__", pd.Timestamp.now().strftime("%d/%m %H:%M")))
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(html)


# ---------- principal ----------
def gerar(pasta):
    garantir_dados(pasta)
    df = carregar(pasta)
    X, fs = montar(df)
    print(f"{len(df)} jogos | {df.Div.nunique()} ligas")

    # teste temporal: treina nos 80% mais antigos, avalia nos 20% mais recentes
    cut = int(len(df) * 0.8)
    stats = {}
    for nome, col in (("over", "y_over"), ("btts", "y_btts")):
        y = df[col].values
        m = modelo().fit(X[:cut], y[:cut])
        p = m.predict_proba(X[cut:])[:, 1]
        base = np.full_like(p, y[:cut].mean())
        print(f"{nome}: Brier {brier_score_loss(y[cut:], p):.4f} (baseline {brier_score_loss(y[cut:], base):.4f})")
        stats[nome] = confiabilidade(y[cut:], p)

    # próximos jogos
    jogos = pasta.rstrip("/") + "_fixtures.csv"
    if not baixar(f"{BASE}/fixtures.csv", jogos):
        sys.exit("Não consegui baixar os próximos jogos (sem internet?).")
    fx = limpar_colunas(pd.read_csv(jogos, encoding="latin-1", on_bad_lines="skip"))
    fx = fx.dropna(subset=["Div", "HomeTeam", "AwayTeam", "Date"]).copy()
    fx["Data"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce", format="mixed")
    fx = fx.dropna(subset=["Data"]).reset_index(drop=True)

    Xf = np.array([fs.row(r.Div, r.HomeTeam, r.AwayTeam) for r in fx.itertuples()], dtype=float)
    fx["P_over"] = modelo().fit(X, df.y_over.values).predict_proba(Xf)[:, 1]
    fx["P_btts"] = modelo().fit(X, df.y_btts.values).predict_proba(Xf)[:, 1]
    for c in ("Avg>2.5", "B365>2.5", "Max>2.5"):
        if c in fx:
            fx["OddCasa"] = pd.to_numeric(fx[c], errors="coerce")
            fx["Edge"] = fx.P_over - 1 / fx.OddCasa
            break

    destino = pasta.rstrip("/") + "_palpites.html"
    gerar_html(fx, stats, destino)
    fx.drop(columns=["Data"]).to_csv(pasta.rstrip("/") + "_palpites.csv", index=False)
    print(f"{len(fx)} jogos analisados.")
    return destino


def servir(pasta, destino, porta=8000):
    """Mini site local: abra http://localhost:8000 no navegador."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/atualizar"):
                try:
                    gerar(pasta)
                except BaseException as e:
                    print("Falha ao atualizar:", e)
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return
            try:
                dados = open(destino, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
            except Exception:
                dados = "Página não encontrada".encode()
                self.send_response(404)
            self.end_headers()
            self.wfile.write(dados)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", porta), H)
    print(f"\n✅ Pronto! Abra no Chrome:  http://localhost:{porta}")
    print("Deixe o Pydroid aberto (não feche esta tela) enquanto usa.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    if "--ci" in sys.argv:  # modo publicação automática (GitHub Actions)
        import shutil
        destino = gerar("dados")
        os.makedirs("site", exist_ok=True)
        shutil.copy(destino, "site/index.html")
        print("Site gerado em site/index.html")
        return
    pasta = next((a for a in sys.argv[1:] if not a.startswith("--")), PASTA)
    destino = gerar(pasta)
    if "--sem-servidor" in sys.argv:
        print(f"Página: {destino}")
        return
    servir(pasta, destino)


if __name__ == "__main__":
    main()
