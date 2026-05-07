"""Measure angles between mean Oja CD-layer category representations."""

from pathlib import Path
import sys
import types

import numpy as np

if __name__ == '__main__' and 'readline' not in sys.modules:
    # This Anaconda Python can segfault importing readline via torch -> pdb.
    sys.modules['readline'] = types.SimpleNamespace(
        set_completer=lambda *args, **kwargs: None,
        parse_and_bind=lambda *args, **kwargs: None,
    )

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental_setups.hebbian_model.oja_variant_model import SimpleHebbianModel as OjaModel

# ── stimulus indices (fixed) ──────────────────────────────────────────
big_animal_indices    = [118, 104,  30,  48, 116]
small_animal_indices  = [105, 115,  29,  57,  59]
animal_indices        = big_animal_indices + small_animal_indices

big_instrument_indices   = [248, 252, 261, 263, 257]
small_instrument_indices = [262, 253, 266, 267, 260]
instrument_indices       = big_instrument_indices + small_instrument_indices

N_ITEMS = 350
ANIMAL_CONTEXT_INDEX = 0
INSTRUMENT_CONTEXT_INDEX = 1
ALPHA_EMA = 0.87


# ── CD rep extractor ──────────────────────────────────────────────────
def make_context_vector(context_index, num_tasks):
    """Build the same one-hot context vector used by the training data."""
    context_vec = torch.zeros(num_tasks)
    context_vec[context_index] = 1.0
    return context_vec


def model_device(model):
    """Return the device of a model even if it has no explicit device attr."""
    return next(model.parameters()).device


def get_cd_reps(model, item_indices, context_vec):
    """
    Returns CD layer activations for each item in item_indices
    given a fixed context vector. Shape: (len(item_indices), cd_dim).
    """
    device = model_device(model)
    item_indices = torch.as_tensor(item_indices, dtype=torch.long, device=device)
    context_vec = context_vec.to(device=device, dtype=torch.float)

    # Do not call model.eval(): SimpleHebbianModel overrides train().
    # These layers are deterministic, and no_grad is enough for this readout.
    with torch.no_grad():
        item_onehot = torch.zeros(
            (len(item_indices), model.num_objects),
            dtype=torch.float,
            device=device,
        )
        item_onehot[torch.arange(len(item_indices), device=device), item_indices] = 1.0
        context = context_vec.unsqueeze(0).repeat(len(item_indices), 1)
        hub_rep = torch.sigmoid(model.item_input_to_hub_weights(item_onehot))
        return torch.sigmoid(
            model.context_input_to_context_dependent_rep_weights(context)
            + model.hub_to_context_dependent_rep_weights(hub_rep)
        )


def load_oja_model(ckpt, alpha_ema=ALPHA_EMA):
    """Load an Oja model checkpoint from a state dict or saved module."""
    model = OjaModel(num_tasks=5, alpha_ema=alpha_ema, device='cpu')
    try:
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location='cpu')
    except Exception:
        state = torch.load(ckpt, map_location='cpu', weights_only=False)

    if isinstance(state, torch.nn.Module):
        return state.to('cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model.load_state_dict(state)
    return model


def angle_between(mean_a, mean_b):
    """Return cosine similarity and angle in degrees between two vectors."""
    cos_sim = F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()
    angle = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
    return cos_sim, angle


def measure_model_angle(model):
    """Measure the animal-vs-instrument CD mean angle for a loaded model."""
    animal_context = make_context_vector(ANIMAL_CONTEXT_INDEX, model.num_tasks)
    instrument_context = make_context_vector(INSTRUMENT_CONTEXT_INDEX, model.num_tasks)

    reps_animal = get_cd_reps(model, animal_indices, animal_context)
    reps_instrument = get_cd_reps(model, instrument_indices, instrument_context)
    return angle_between(reps_animal.mean(dim=0), reps_instrument.mean(dim=0))


def measure_angles(model_dir=Path('models'), n_seeds=10, alpha_ema=ALPHA_EMA):
    results = {'blocked': [], 'interleaved': []}

    for curriculum, tag in [('blocked', 'blockd'), ('interleaved', 'interl')]:
        for seed in range(n_seeds):
            ckpt = Path(model_dir) / f'oja_a{alpha_ema:.2f}_{tag}-{seed}.torch'
            if not ckpt.exists():
                raise FileNotFoundError(f'Missing checkpoint: {ckpt}')

            model = load_oja_model(ckpt, alpha_ema=alpha_ema)
            cos_sim, angle = measure_model_angle(model)

            results[curriculum].append(angle)
            print(f'  {curriculum} seed {seed}: angle={angle:.2f} deg  cos={cos_sim:.4f}')

    return results


def print_summary(results):
    print()
    for curriculum, angles in results.items():
        arr = np.array(angles)
        print(f'{curriculum:>12}:  angle = {arr.mean():.2f} +/- {arr.std():.2f} deg'
              f'   (mean cos = {np.cos(np.radians(arr)).mean():.4f})')


def main():
    results = measure_angles()
    print_summary(results)


if __name__ == '__main__':
    main()
