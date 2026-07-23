import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
// 1. Import the router
import { BrowserRouter } from 'react-router-dom'; 
import App from './App.tsx';

// @ts-ignore
import './index.css'; 

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found');
}

createRoot(rootElement).render(
  <StrictMode>
    {/* 2. Wrap App inside the router! */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);