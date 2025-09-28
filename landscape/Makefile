PYTHON?=python3

.PHONY: test install

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Run landscape test suite (requires Supabase credentials for integration tests)
test:
	pytest tests
