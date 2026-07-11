# Frame_Dumper — Spec d'intégration

**Projet** : Rl-nameblur · **Date** : 2026-07-10 · **Statut** : ✅ Verrouillé sur codebase réelle — prêt pour génération

---

## 1. Objectif

Dumper sur disque toute frame dont le `frame_id` apparaît dans le flux **events** (couvre détection SLOW + lifecycle FAST), plus `N` frames de tail configurables.

**Formule de sélection :**

```text
frames_à_dumper = { frame_id ∈ events | frame_id != -1 }
                ∪ { f+1 … f+N | f ∈ events, f != -1, N = tail_frames }
```

- **Filtre `!= -1` obligatoire** : `Detection`, `Mask` et `LifecycleRecord` ont tous `frame_id` par défaut `-1`. Un event fast-only émis avant ancrage slow sort avec `-1` → **skippé silencieusement** (cohérent avec la politique « pas d'erreur »).
- Dédup obligatoire sur l'union.
- Tail absent si `f+1` non capturée — pas d'erreur.

---

## 2. Chaînage `frame_id` — en place

Le `frame_id` est propagé de bout en bout dans la codebase. **Zéro modification de struct.**

| Maillon                                                                      | Statut |
| ---------------------------------------------------------------------------- | ------ |
| `main.py → Detection(frame_id=frame_id)`                                     | ✅     |
| `Detection.frame_id: int = -1` (`models.py`)                                 | ✅     |
| `MaskRegistry.create(frame_id=det.frame_id)` → `Mask.frame_id` (défaut `-1`) | ✅     |
| `emit_lifecycle` : `getattr(mask, "frame_id", -1)` → `record["frame_id"]`    | ✅     |
| `LifecycleRecord.frame_id: int`                                              | ✅     |

Seule action liée : **validation défensive** `frame_id != -1` au dump.

---

## 3. Arbitrages figés

| Paramètre                         | Valeur                                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Format                            | JPEG, **Q75** par défaut, configurable                                                                          |
| Nommage                           | `frame_{session_id}_{frame_id}.jpg`                                                                             |
| `session_id`                      | issu du singleton `bench._session_id` (format `debug.bench.writer.session_id_format`)                           |
| `tail_frames`                     | `0` (couvertes seules) / `2` (étendu chiffré)                                                                   |
| Ancrage                           | flux `events`                                                                                                   |
| Ring buffer                       | créé sur `CaptureThread` (frame source non altérée)                                                             |
| **Dimensionnement `ring_size`**   | **Stratégie B — conservateur** : `seuil_survie_tracker_max_frames + tail_frames + marge`, borné par plafond RAM |
| **Politique de saturation queue** | **DROP** (`put_nowait` + `except Full`) — jamais bloquant                                                       |
| Écriture                          | async `queue + thread` daemon, calqué sur `BenchJsonlWriter`                                                    |

---

## 4. Architecture d'intégration

### 4.1 Ring buffer (`CaptureThread`)

- Structure : `collections.deque(maxlen=ring_size)` de tuples `(frame_id, frame_copy)`.
- **Copie obligatoire** à l'insertion (les buffers `sender.borrow()` sont recyclés — référence non sûre).
- **Dimensionnement — Stratégie B (conservateur)** :

  ```text
  ring_size ≥ seuil_survie_tracker_max_frames + tail_frames + marge (~2×)
  ```

  Calé sur le plus grand seuil de survie du tracker (driver du pire cas : events terminaux `LOST`/`EXPIRED`/`EVICTED` référençant une frame ancienne), converti à la cadence de capture. Robuste sans calibration préalable ; affinable ensuite par mesure.

- **Coût RAM** : 1 frame 1080p BGR ≈ 6,2 MB → 240 frames ≈ 1,5 GB. `ring_size` **borné par un plafond RAM explicite**.
- **Filet de sécurité** : `frame_id` absent du ring → skip silencieux + `ring_miss_count++` + `log.warning` throttlé (signal de sous-dimensionnement).

### 4.2 `FrameDumperWriter` (calqué sur `BenchJsonlWriter`)

- Squelette identique : `__init__ / start / stop / _write_loop`, `queue.Queue(maxsize=queue_maxsize)`, thread daemon, sentinelle à l'arrêt.
- Diffère par : encodage JPEG (`cv2.imencode('.jpg', frame, [IMWRITE_JPEG_QUALITY, q])`) ; chemin `frame_dumper.path` ; dédup union+tail en amont de l'enqueue.
- **Politique de saturation — DROP** :

  ```python
  try:
      self._queue.put_nowait(item)
  except queue.Full:
      self._drop_count += 1   # + log.warning throttlé
  ```

  Le dump est du debug, hors chemin critique : jamais de backpressure sur capture/traitement.

- **Découplage** : consomme le flux events + interroge le ring buffer par `frame_id`. Si absent → skip sans erreur.

### 4.3 Observabilité — 3 compteurs distincts

| Compteur                | Cause                              | Levier                                      |
| ----------------------- | ---------------------------------- | ------------------------------------------- |
| `ring_miss_count`       | frame évincée avant l'event        | ↑ `ring_size`                               |
| `queue_drop_count`      | queue saturée (I/O/encodage lents) | ↑ `queue_maxsize`, disque, ↓ `jpeg_quality` |
| `skip_invalid_frame_id` | `frame_id == -1`                   | attendu, informatif                         |

### 4.4 Impact `main.py` / modules métier

- **Nul** au-delà de l'instanciation du writer et de l'alimentation du ring buffer dans `CaptureThread`. Aucun struct modifié.

---

## 5. Config — section `frame_dumper` à ajouter

```yaml
frame_dumper:
  enabled: false
  path: "..." # répertoire de sortie
  jpeg_quality: 75 # configurable
  tail_frames: 0 # 0 | 2
  ring_size: 240 # Stratégie B : survie_tracker_max + tail + marge, borné RAM
  queue_maxsize: 256 # drop au-delà (jamais bloquant)
  saturation_policy: drop # figé
```

---

## 6. Chiffrage de référence

Session `210315` · 43,6 s · 1920×1080 · 120 fps · span 5712 · 156 frames couvertes, toutes isolées.

| tail | Frames | % span | JPEG Q75 session | ~1 h extrapolé |
| ---- | -----: | -----: | ---------------: | -------------- |
| `0`  |    156 |  2,7 % |            65 MB | ~2,1 GB        |
| `2`  |    468 |  8,2 % |           194 MB | ~6,3 GB        |

B étendu +2 = ÷12,2 vs « toutes frames ». PNG écarté (~17 GB/h).

---

## 7. Points ouverts restants (non bloquants)

- [x] **Dimensionnement `ring_size`** → **Stratégie B (conservateur)**, borné RAM, affiné par mesure ultérieure.
- [x] **Politique de saturation** → **DROP** + `queue_drop_count` + warning throttlé.
- [x] **Nettoyage cosmétique** : doublon `frame_id: int` dans `LifecycleRecord` (`lifecycle.py`) — à retirer.
- [x] **Comportement `frame_id == -1`** → **skip silencieux** + `skip_invalid_frame_id`.

**Valeurs à finaliser à l'usage** : `ring_size` exact (calage sur le seuil de survie tracker réel) et `queue_maxsize` (calage sur pic d'events observé) — ajustables via les 3 compteurs, sans changement de code.

---

## 8. Périmètre non relu (transparence)

- `TrackerConfig` (`models.py`) non relu intégralement — **hors périmètre**, ne contient pas de `frame_id`.
- Signatures `bench.push_frame/push_events/push_detections` : présentes, réutilisées telles quelles.

La spec est verrouillée sur le code réel.

---
