build_db:
	docker-compose up -d
	sleep 5
	python setup/insert_data.py

build_enhanced_db:
	docker-compose up -d
	sleep 5
	echo "Generating synthetic examples using OpenAI..."
	echo "Warning: This will take a long time to run and may be expensive."
	python setup/insert_enhanced_data.py

destroy_db:
	@read -p "Are you sure you want to destroy the database? [y/N] " -n 1 -r; \
	echo ; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down; \
		sleep 5; \
		rm -rf postgres_data; \
	fi

start_db:
	docker-compose up -d

stop_db:
	docker-compose down