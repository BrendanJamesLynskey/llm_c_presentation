// ========== llm.c — interactive demos ==========
// Every getElementById is guarded with `if (!el) return;`

// ========== Demo 1: Kernel Sequence Stepper ==========
// Walks the forward -> zero_grad -> backward -> update sequence,
// highlighting each kernel as it runs.
const KERNEL_STEPS = [
    { name: 'encoder_forward',          phase: 'fwd',
      desc: 'FORWARD. Token + position embeddings: out = wte[token] + wpe[pos]. The input ids become C-dimensional vectors.' },
    { name: 'layernorm_forward',        phase: 'fwd',
      desc: 'FORWARD. Normalise each token to zero mean / unit variance, then scale and shift. mean & rstd are cached for backward.' },
    { name: 'matmul_forward (qkv)',     phase: 'fwd',
      desc: 'FORWARD. One fused linear layer produces Q, K and V together: (B,T,C) -> (B,T,3C).' },
    { name: 'attention_forward',        phase: 'fwd',
      desc: 'FORWARD. q.k^T, scale by 1/sqrt(hs), causal mask (t2<=t), softmax, then the weighted sum over values.' },
    { name: 'matmul_forward (proj)',    phase: 'fwd',
      desc: 'FORWARD. Project the attention output back to C, then residual_forward adds it to the block input.' },
    { name: 'matmul + gelu + matmul',   phase: 'fwd',
      desc: 'FORWARD. The MLP: up-project to 4C, GELU non-linearity, down-project to C, residual add. (Repeats for every layer.)' },
    { name: 'softmax + crossentropy',   phase: 'fwd',
      desc: 'FORWARD. Final layernorm, classifier matmul to logits, softmax, then the mean negative-log-likelihood loss.' },
    { name: 'gpt2_zero_grad',           phase: 'upd',
      desc: 'ZERO. memset(grads_memory, 0): clear the gradient buffer (and activation grads) before accumulating fresh gradients.' },
    { name: 'crossentropy_softmax_backward', phase: 'bwd',
      desc: 'BACKWARD. Seed the gradient chain with dloss; the softmax+CE gradient is simply (probs - onehot(target)).' },
    { name: 'matmul_backward',          phase: 'bwd',
      desc: 'BACKWARD. One forward matmul -> three grads: dinp, dweight (+= dout^T @ inp) and dbias. Runs for head, MLP and qkv.' },
    { name: 'gelu / residual_backward', phase: 'bwd',
      desc: 'BACKWARD. Undo the MLP non-linearity and split the residual gradient back into both of its branches.' },
    { name: 'attention_backward',       phase: 'bwd',
      desc: 'BACKWARD. Differentiate softmax and the two matmuls (q.k^T and att.v) to recover gradients on q, k and v.' },
    { name: 'layernorm_backward',       phase: 'bwd',
      desc: 'BACKWARD. The hand-derived layernorm gradient, reusing cached mean/rstd. Note += : dweight/dbias accumulate over B,T.' },
    { name: 'encoder_backward',         phase: 'bwd',
      desc: 'BACKWARD. Scatter gradients back into the wte and wpe embedding tables. The reversed chain is complete.' },
    { name: 'gpt2_update (AdamW)',      phase: 'upd',
      desc: 'UPDATE. One flat loop over every parameter: update m & v moments, bias-correct, apply decoupled weight decay, step.' },
];

let kernelIdx = 0;
let kernelTimer = null;

function renderKernelTrack() {
    const track = document.getElementById('kernel-track');
    if (!track) return;
    track.innerHTML = '';
    KERNEL_STEPS.forEach((s, i) => {
        const node = document.createElement('div');
        node.className = 'kernel-node phase-' + s.phase + (i === kernelIdx ? ' active' : '');
        node.textContent = s.name;
        node.addEventListener('click', () => { setKernelStep(i); });
        track.appendChild(node);
    });
}

function setKernelStep(i) {
    const desc = document.getElementById('kernel-desc');
    if (!desc) return;
    kernelIdx = (i + KERNEL_STEPS.length) % KERNEL_STEPS.length;
    desc.textContent = KERNEL_STEPS[kernelIdx].desc;
    document.querySelectorAll('#kernel-track .kernel-node').forEach((n, j) => {
        n.classList.toggle('active', j === kernelIdx);
    });
}

function stopKernelPlay() {
    if (kernelTimer) { clearInterval(kernelTimer); kernelTimer = null; }
    const playBtn = document.getElementById('kernel-play');
    if (playBtn) playBtn.textContent = 'Play ▶';
}

document.addEventListener('DOMContentLoaded', () => {
    const track = document.getElementById('kernel-track');
    if (!track) return;
    renderKernelTrack();
    setKernelStep(0);

    const prev = document.getElementById('kernel-prev');
    const next = document.getElementById('kernel-next');
    const play = document.getElementById('kernel-play');

    if (prev) prev.addEventListener('click', () => { stopKernelPlay(); setKernelStep(kernelIdx - 1); });
    if (next) next.addEventListener('click', () => { stopKernelPlay(); setKernelStep(kernelIdx + 1); });
    if (play) play.addEventListener('click', () => {
        if (kernelTimer) { stopKernelPlay(); return; }
        play.textContent = 'Pause ⏸';
        kernelTimer = setInterval(() => {
            if (kernelIdx === KERNEL_STEPS.length - 1) { stopKernelPlay(); return; }
            setKernelStep(kernelIdx + 1);
        }, 1100);
    });
});

// ========== Demo 2: GPT-2 Memory Budget Calculator ==========
// params, AdamW optimizer state (m + v), and a rough activation estimate.
const GPT2_SIZES = {
    // params = approximate trainable parameter count
    // L, C, T used for a rough activation-memory estimate
    '124M':  { params: 124e6,  L: 12, C: 768,  heads: 12 },
    '350M':  { params: 350e6,  L: 24, C: 1024, heads: 16 },
    '774M':  { params: 774e6,  L: 36, C: 1280, heads: 20 },
    '1558M': { params: 1558e6, L: 48, C: 1600, heads: 25 },
};

const BYTES_FP32 = 4;
const GB = 1024 * 1024 * 1024;

function fmtGB(bytes) {
    const g = bytes / GB;
    if (g >= 10) return g.toFixed(1) + ' GB';
    return g.toFixed(2) + ' GB';
}

function renderBudget(sizeKey) {
    const bars = document.getElementById('budget-bars');
    const note = document.getElementById('budget-note');
    if (!bars) return;

    const cfg = GP_T2_SIZE_LOOKUP(sizeKey);
    if (!cfg) return;

    // fp32 master copies, as in the CPU reference / fp32 CUDA build
    const paramBytes = cfg.params * BYTES_FP32;       // params_memory
    const gradBytes  = cfg.params * BYTES_FP32;       // grads_memory (mirror)
    const optimBytes = cfg.params * 2 * BYTES_FP32;   // AdamW m + v

    // rough activation estimate for one batch (B=4, T=1024):
    //   dominated by ~ B * T * C * L floats of stored activations
    const B = 4, T = 1024;
    const actBytes = B * T * cfg.C * cfg.L * 4 * BYTES_FP32; // ~4 big tensors/layer

    const total = paramBytes + gradBytes + optimBytes + actBytes;
    const max = total; // scale bars to the total

    const rows = [
        { label: 'Parameters',        bytes: paramBytes, cls: 'bf-params' },
        { label: 'Gradients',         bytes: gradBytes,  cls: 'bf-grads'  },
        { label: 'AdamW state (m+v)', bytes: optimBytes, cls: 'bf-optim'  },
        { label: 'Activations (B4·T1024)', bytes: actBytes, cls: 'bf-total' },
    ];

    bars.innerHTML = '';
    rows.forEach(r => {
        const row = document.createElement('div');
        row.className = 'budget-bar-row';

        const label = document.createElement('div');
        label.className = 'budget-bar-label';
        label.textContent = r.label;

        const track = document.createElement('div');
        track.className = 'budget-bar-track';

        const fill = document.createElement('div');
        fill.className = 'budget-bar-fill ' + r.cls;
        const pct = Math.max(2, (r.bytes / max) * 100);
        fill.style.width = pct + '%';
        fill.textContent = fmtGB(r.bytes);

        track.appendChild(fill);
        row.appendChild(label);
        row.appendChild(track);
        bars.appendChild(row);
    });

    if (note) {
        const paramsM = Math.round(cfg.params / 1e6);
        note.innerHTML = 'GPT-2 <strong>' + paramsM + 'M</strong> &middot; ' +
            cfg.L + ' layers &middot; C=' + cfg.C + ' &middot; ' + cfg.heads + ' heads. ' +
            'Total (fp32, this estimate): <strong>' + fmtGB(total) + '</strong>. ' +
            'AdamW alone needs 2&times; the parameter memory (m and v); a BF16 path roughly halves the weight/activation cost.';
    }
}

// small helper so a typo can't silently break the lookup
function GP_T2_SIZE_LOOKUP(key) {
    return GPT2_SIZES[key] || null;
}

document.addEventListener('DOMContentLoaded', () => {
    const sizes = document.getElementById('budget-sizes');
    if (!sizes) return;

    sizes.querySelectorAll('.budget-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            sizes.querySelectorAll('.budget-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderBudget(btn.dataset.size);
        });
    });

    renderBudget('124M');
});
