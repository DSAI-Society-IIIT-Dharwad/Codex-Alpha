import { useState } from 'react'
import { Sidebar, Hero} from './components/index.js';
import './App.css'


function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="App">
      <Sidebar />
      <Hero />
    </div>
  )
}

export default App
