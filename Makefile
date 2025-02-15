build_db:
	docker-compose up -d
	sleep 5
	python setup/insert_data.py

destroy_db:
	docker-compose down
	sleep 5
	rm -rf postgres_data

start_db:
	docker-compose up -d

stop_db:
	docker-compose down