### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 26657c80-3523-4756-ab30-cf0bd552070d
using Pkg; Pkg.add("Impute")

# ╔═╡ 04e03fd4-63e5-4b54-a4ea-21927e8cd5cb
using DelimitedFiles, DataFrames, CategoricalArrays, impute 

# ╔═╡ 5f7a7820-5c1e-11ee-1767-5f554e5a74b9
begin
	switzerland_t_data = readdlm("processed.switzerland.data", ',', header=false)
	raw_switzerland_data = DataFrame(switzerland_t_data, :auto)
	named_switzerland = rename(raw_switzerland_data, :x1 => :"Age", :x2 => :"Sex", :x3 => :"Chest Pain Type", :x4 => :"Resting Blood Pressure", :x5 => :"Serum Cholesterol in mg/dl", :x6 => :"Fasting Blood Sugar > 120 mg/dl", :x7 => :"Resting Electrocaridographic Results", :x8 => :"Maximum Heart Rate Achieved", :x9 => :"Exercise Induced Angina", :x10 => :"Oldpeak = ST Depression Induced by Exercuse Relative to Rest", :x11 => :"Slope of the Peak Exercise ST Segment", :x12 => :"Number of Major Vessels colored by flourosopy", :x13 => :"Thal: 3 = Normal; 6 = Fixed Defect; 7 = Reversable Defect", :x14 => :"Presence of heart disease")
end;

# ╔═╡ cd5b9656-71d5-4de9-9ca4-39db0c5bbe4b
begin
	hungarian_t_data = readdlm("processed.hungarian.data", ',', header=false)
	raw_hungarian_data = DataFrame(hungarian_t_data, :auto)
	named_hungarian = rename(raw_hungarian_data, :x1 => :"Age", :x2 => :"Sex", :x3 => :"Chest Pain Type", :x4 => :"Resting Blood Pressure", :x5 => :"Serum Cholesterol in mg/dl", :x6 => :"Fasting Blood Sugar > 120 mg/dl", :x7 => :"Resting Electrocaridographic Results", :x8 => :"Maximum Heart Rate Achieved", :x9 => :"Exercise Induced Angina", :x10 => :"Oldpeak = ST Depression Induced by Exercuse Relative to Rest", :x11 => :"Slope of the Peak Exercise ST Segment", :x12 => :"Number of Major Vessels colored by flourosopy", :x13 => :"Thal: 3 = Normal; 6 = Fixed Defect; 7 = Reversable Defect", :x14 => :"Presence of heart disease")
end;

# ╔═╡ def0ad95-c2e2-4916-860f-6129df3fbcc2
begin
	va_t_data = readdlm("processed.va.data", ',', header=false)
	raw_va_data = DataFrame(va_t_data, :auto)
	named_va = rename(raw_va_data, :x1 => :"Age", :x2 => :"Sex", :x3 => :"Chest Pain Type", :x4 => :"Resting Blood Pressure", :x5 => :"Serum Cholesterol in mg/dl", :x6 => :"Fasting Blood Sugar > 120 mg/dl", :x7 => :"Resting Electrocaridographic Results", :x8 => :"Maximum Heart Rate Achieved", :x9 => :"Exercise Induced Angina", :x10 => :"Oldpeak = ST Depression Induced by Exercuse Relative to Rest", :x11 => :"Slope of the Peak Exercise ST Segment", :x12 => :"Number of Major Vessels colored by flourosopy", :x13 => :"Thal: 3 = Normal; 6 = Fixed Defect; 7 = Reversable Defect", :x14 => :"Presence of heart disease")
end;

# ╔═╡ Cell order:
# ╠═04e03fd4-63e5-4b54-a4ea-21927e8cd5cb
# ╠═26657c80-3523-4756-ab30-cf0bd552070d
# ╠═5f7a7820-5c1e-11ee-1767-5f554e5a74b9
# ╠═cd5b9656-71d5-4de9-9ca4-39db0c5bbe4b
# ╠═def0ad95-c2e2-4916-860f-6129df3fbcc2
