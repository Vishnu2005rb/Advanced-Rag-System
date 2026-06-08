import React from "react";
import { motion } from "framer-motion";

interface StatsCardProps {
  value: string | number;
  label: string;
  icon: React.ReactNode;
  color: "blue" | "gold" | "green" | "purple" | "cyan";
}

export default function StatsCard({ value, label, icon, color }: StatsCardProps) {
  const colorMap = {
    blue: {
      text: "text-blue-400",
      bg: "bg-blue-500/10 border-blue-500/20",
      iconColor: "text-blue-400",
    },
    gold: {
      text: "text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/20",
      iconColor: "text-amber-400",
    },
    green: {
      text: "text-emerald-400",
      bg: "bg-emerald-500/10 border-emerald-500/20",
      iconColor: "text-emerald-400",
    },
    purple: {
      text: "text-purple-400",
      bg: "bg-purple-500/10 border-purple-500/20",
      iconColor: "text-purple-400",
    },
    cyan: {
      text: "text-cyan-400",
      bg: "bg-cyan-500/10 border-cyan-500/20",
      iconColor: "text-cyan-400",
    },
  };

  const currentColors = colorMap[color] || colorMap.blue;

  return (
    <motion.div
      whileHover={{ y: -2, borderColor: "rgba(255,255,255,0.15)" }}
      transition={{ duration: 0.2 }}
      className="flex items-center gap-3 bg-slate-900/80 border border-slate-800/80 rounded-xl p-3 shadow-lg"
    >
      <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${currentColors.bg} ${currentColors.iconColor}`}>
        {icon}
      </div>
      <div className="flex flex-col">
        <span className={`font-bold text-sm leading-tight ${currentColors.text}`}>
          {value}
        </span>
        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          {label}
        </span>
      </div>
    </motion.div>
  );
}
