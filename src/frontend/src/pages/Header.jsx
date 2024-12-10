import React from "react";
import { motion } from "motion/react";
import Statistics from "./Statistics";

const Header = () => {
	return (
        <div className="h-[60px] border-b-[1px] mx-5 flex items-center">
            <div className="w-full flex justify-between">
                <div className="flex items-center">
                    <img src="/logo192.png" className="w-[50px]" />
                    <h1 className="text-lg font-bold">BPM Visualizer</h1>
                </div>
                <Statistics />
                {/* <motion.div 
                    className="flex items-center px-3 py-1 rounded-lg"
                    initial={{
                        backgroundColor: "transparent"
                    }}
                    whileHover={{
                        backgroundColor: "#e9eff2",
                        transition: {
                            duration: 1
                        }
                    }}
                >
                    <a className="text-md cursor-pointer text-gray-600">
                        History
                    </a>
                </motion.div> */}
            </div>
        </div>
    );
};

export default Header;