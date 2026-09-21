SELECT 'CREATE DATABASE selfsuvis_fusion'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'selfsuvis_fusion')\gexec
