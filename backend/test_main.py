from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)
    
def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"Status": "l'API fonctionne parfaitement !"}
    

