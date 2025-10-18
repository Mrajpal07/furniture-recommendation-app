import { BrowserRouter, Routes, Route } from 'react-router-dom';
import TestComponent from './components/TestComponent';

function App() {
  return (
    <BrowserRouter>
      <div className="container mx-auto p-4">
        <Routes>
          <Route path="/" element={
            <>
              <h1 className="text-2xl">Home</h1>
              <div className="bg-blue-500 text-white p-4">Test Tailwind</div>
              <TestComponent />
            </>
          } />
          <Route path="/recommend" element={<h1 className="text-2xl">Recommendations</h1>} />
          <Route path="/analytics" element={<h1 className="text-2xl">Analytics</h1>} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;