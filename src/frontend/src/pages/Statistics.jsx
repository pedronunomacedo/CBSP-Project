import React, { useState } from "react";
import { Button, Modal } from "antd";
import Table from "../components/Table";

const Statistics = () => {
	const [isModalOpen, setIsModalOpen] = useState(false);

	const showModal = () => {
		setIsModalOpen(true);
	};

	const handleCancel = () => {
		setIsModalOpen(false);
	};

    return (
        <>
            <div
                className="flex items-center"
            >
                <Button 
                    className="w-full h-full text-md cursor-pointer text-gray-600"
                    onClick={showModal}
                >
                    History
                </Button>
            </div>
            <Modal
                title="History"
                open={isModalOpen}
                onCancel={handleCancel}
                width={1000}
                footer={[]}
            >
                <Table />
            </Modal>
        </>
    );
};

export default Statistics;
