import { Link } from "react-router-dom"
import "./Header.css"


function Header({name, username}: {name: string, username: string}){

    return (
        <div className="header-container">
            <Link to='/' className="header-title">
                {name}
            </Link>
            <div className="user">
                {username}
            </div>
        </div>

    )
}

export default Header