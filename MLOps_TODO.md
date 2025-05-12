# 📌 MLOps WeatherAUS – Project Tracker

A living checklist and roadmap to guide the MLOps WeatherAUS repo toward production readiness.

---

## ✅ PHASE 0: Project Setup

- [x] Create directory structure based on wine repo
- [x] Push all changes to GitHub
- [x] Clean and update `requirements.txt`
- [ ] Add project overview in `README.md`
- [ ] Save this tracker as `MLOps_TODO.md`

---

## ⚙️ PHASE 1: Base Configuration

- [ ] Copy and adapt `Makefile`
- [ ] Add `Dockerfile` based on wine repo
- [ ] Set up GitHub Actions workflow (`.github/workflows/main.yml`)
- [ ] Add `.dvc` setup (for data tracking)
- [ ] Add `setup.py` with package info

---

## 🧪 PHASE 2: Notebooks & Experiments

- [ ] Move Jupyter notebooks to `/notebooks/`
- [ ] Confirm EDA, preprocessing, and modeling are working
- [ ] Summarize experiments in a README inside `/notebooks/`
- [ ] Create chunks of `weatherAUS.csv` for post-deployment testing

---

## 🧠 PHASE 3: Modular Code in `/src`

- [ ] Create data loading script in `src/data/`
- [ ] Add feature engineering to `src/features/`
- [ ] Add model training and prediction to `src/models/`
- [ ] Add reusable utilities in `src/utils/`
- [ ] Rebuild `train.py`, `preprocessing.py`, and `train_model.py` for modular code

---

## 🧪 PHASE 4: Testing

- [ ] Set up Pytest in `/tests/`
- [ ] Add basic tests for each module
- [ ] Add test data if needed

---

## 🛰️ PHASE 5: API and Deployment

- [ ] Create FastAPI app for predictions
- [ ] Add `main.py` and prediction endpoint
- [ ] Test locally with `uvicorn`
- [ ] Prepare for deployment (e.g. Docker)

---

## 🚀 Final Polish

- [ ] Final README with badges, architecture, usage
- [ ] Clean up unused files
- [ ] Create release version/tag

---

✍️ **Notes**
- If errors or issues arise, document them here with resolutions.
- Keep track of important links, configs, or dataset versions.

