import Header from "./Header"
import { useState } from "react";
import type { Message } from "./types/Message";
import "./Home.css";

function Home (){
    let name = "UFABChatBot"
    let username = "Arari"
    let welcome = "Bem vindo ao UFABChatBot!"
    const [prompt, setPrompt] = useState("")
    const [messages, setMessages] = useState<Message[]>([])

    function newPrompt(event: React.ChangeEvent<HTMLInputElement>){
        setPrompt(event.target.value)
    }

    function enter(event: React.KeyboardEvent<HTMLInputElement>){
        if (event.key === "Enter") {
            handlePrompt()
        }
    }

    async function handlePrompt (){
        let send_message:Message = {
            id: crypto.randomUUID(),
            text: prompt,
            sender: "user",
            sentAt: new Date()
        }

        let all_messages = [...messages, send_message]
        setMessages(all_messages)
        setPrompt("")

        const response = await fetch ("http://localhost:8000/prompt", {
            method: "POST",
            body: JSON.stringify({
                prompt: prompt
            }),
            headers: {"Content-Type": "application/json"}
        })

        if (!response.ok) {
            alert("Failed to get response")
            return "fail"
        }

        const data = await response.json()

        let response_message:Message = {
            id: crypto.randomUUID(),
            text: data.text,
            sender: 'bot',
            sentAt: new Date()
        }
 
        setMessages((current) => [...current, response_message])
        return response
    }

    return (
        <div className="home-container">
            <h1>
                <Header 
                name = {name}
                username = {username}
                />
            </h1>
            <p> {welcome} </p>
            <div className="chat-box">
                {messages.map( (msg) => (
                    <div key={msg.id} className={msg.sender}>
                        {msg.text}
                    </div>
                ))}
            </div>
            <div className="prompt-container">
                <input className="prompt-box" 
                    type ="text" 
                    placeholder="Qual sua dúvida?" 
                    value={prompt} 
                    onChange={newPrompt}
                    onKeyDown={enter}
                />
                <button className="button" onClick = {handlePrompt}></button>
            </div>
        </div>

    )
}

export default Home