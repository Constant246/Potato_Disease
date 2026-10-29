import React, {useState} from 'react';
import axios from 'axios';
import './App.css';

// Icones simple (en SVG) pour un visuel professionel
const UploadIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
  strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v4" /><polyline 
  points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15"/> </svg>
);

const SearchIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#718792" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
);

const BellIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#718792" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
);

const FileIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a3b1c6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
);

const LeafIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#27ae60" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>
);

const LockIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a3b1c6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
);

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [description, setDescription] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('diagnose');

  // Gestion de la sélection de l'image
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  // Envoi de l'image vers l'API FastAPI
  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(true);
    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {'Content-Type': 'multipart/form-data'},
      });
      setResult(response.data);
    } catch (error) {
      console.error("Erreur lors de la prédiction :", error);
      alert("Impossible de contacter l'API FastAPI. Vérifier qu'elle est lancée sur le port 8000 !");
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className='app-viewport'>

      {/* BARRE DE NAVIGATION SUPERIEURE */}
      <header className='top-navbar'>
        <div className='brand-logo'>PotatoPal AI</div>

        <div className='nav-actions'>
          <button className='nav-icon-btn'><SearchIcon /> </button>
          <button className='nav-icon-btn potato-btn'>🥔</button>
          <button className='nav-icon-btn'><BellIcon /> </button>
        </div>
      </header>

      {/* CONTENU PRINCIPAL */}
      <main className='main-container'>

        <div className='title-block'>
          <h1>Identify Potato Disease</h1>
          <p>Upload an image of your potato leaf</p>
        </div>

        {/* CARTE NEUMORPHIQUE CENTRALE */}
        <div className='neumorphic-card'>

          {/* Bouton Sélection de Fichier */}
          <div className='field-group'>
            <label htmlFor='file-upload' className='inset-input-btn'>
              <span className='left-icon'><FileIcon /> </span>
              <span className='file-label'>
                {selectedFile ? selectedFile.name : "Choose File"}
              </span>
              <span className='right-icon'><LeafIcon /> </span>
            </label>
            <input id='file-upload' type='file' accept='image/*' onChange={handleFileChange} />
          </div>

          {/* Champ Description Optionnel */}
          <div className='field-group'>
            <div className='inset-text-input'>
              <span className='left-icon'><LockIcon /> </span>
              <input type='text' placeholder='Description (Optionel)' 
              value={description} onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </div>

          {/* Aperçu d'image une fois sélectionnée */}
          {preview && (
            <div className='preview-box'>
              <img src={preview} alt='Aperçu' />
            </div>
          )}

          {/* Bouton de diagnostic vert */}
          <button onClick={handleUpload}
          disabled={!selectedFile || loading}
          className='green-action-btn'>
            {loading ? "ANALYZING..." : "RUN DIAGNOSIS"}
          </button>

          {/* Résultat du diagnostic */}
          {result && (
            <div className='result-box'>
              <p><strong>Pathologie :</strong> {result.class} </p>
              <p><strong>Confiance :</strong> {result.confidence} </p>
            </div>
          )}

        </div>
        <p className='account-link'>New to PotatoPal? <span>Create an Account</span></p>

      </main>

      {/* BARRE DE NAVIGATION INFERIEURE */}
      <nav className='bottom-navbar'>
        <button className={`nav-tab ${activeTab === 'home' ? 'active' : ''}`}
        onClick={() => setActiveTab('home')}>
          <span className='tab-icon'>🏠</span>
          <span className='tab-label'>Home</span>
        </button>
        <button className={`nav-tab ${activeTab === 'diagnose' ? 'active' : ''}`}
        onClick={() => setActiveTab('diagnose')}
        >
          <span className='tab-icon'>🌿</span>
          <span className='tab-label'>Diagnose</span>
        </button>
        <button className={`nav-tab ${activeTab === 'scan' ? 'active' : ''}`}
        onClick={() => setActiveTab('scan')}
        >
          <span className='tab-icon'>📷</span>
          <span className='tab-label'>Scan</span>
        </button>
        <button className={`nav-tab ${activeTab === 'library' ? 'active' : ''}`}
        onClick={() => setActiveTab('library')}
        >
          <span className='tab-icon'>📚</span>
          <span className='tab-label'>Library</span>
        </button>
        <button className={`nav-tab ${activeTab === 'profile' ? 'active' : ''}`}
        onClick={() => setActiveTab('profile')}
        >
          <span className='tab-icon'>👤</span>
          <span className='tab-icon'>Profile</span>
        </button>
      </nav>

    </div>
  );
}

export default App;
