### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 783722b3-7252-416f-bf1c-f1c6e89e3b40
import Pkg

# ╔═╡ d5f5b922-5166-4ead-9fa7-4836a9ab6fe6
begin 
	Pkg.add("PlotlyJS")
	Pkg.add("MLJMultivariateStatsInterface")
	Pkg.add("CSV")
	Pkg.add("StatsBase")
end;

# ╔═╡ 0797d579-8e97-4b79-94f9-b4249609ab4a
using Plots, PlutoUI, HypertextLiteral, DelimitedFiles, DataFrames, CategoricalArrays, Statistics, Distances, StatsPlots, Compose, ColorSchemes, Distributions

# ╔═╡ 5ca26dc5-0d67-413d-96d1-6fff9e6639eb
using StatsBase, MLJBase, MLJDecisionTreeInterface, Random, DecisionTree, MLJ, TreeRecipe, StableRNGs, BetaML, StatsBase, PlotlyJS, CSV

# ╔═╡ 42196ee4-99a6-4d54-aa33-ddb7a2591392
using LinearAlgebra

# ╔═╡ b4cdec68-0599-4800-a7dc-1a7cb3cd0fd5
begin 
	data1 = DataFrame(readdlm("imputed_va.csv", ',', header=false), :auto);
	data2 = DataFrame(readdlm("imputed_swiss.csv", ',', header=false), :auto);
	data3 = DataFrame(readdlm("imputed_hungarian.csv", ',', header=false), :auto);

	idata = vcat(data1[2:end,:],data2[2:end,:],data3[2:end,:])

	idata_copy = copy(idata)
	
		idata = coerce(idata, 
	:x1      => MLJBase.Continuous,
	:x2      => Multiclass,
	:x3      => Multiclass,
	:x4      => MLJBase.Continuous,
	:x5      => MLJBase.Continuous,
	:x6      => Multiclass,
	:x7      => Multiclass,
	:x8      => MLJBase.Continuous,
	:x9      => Multiclass,
	:x10     => MLJBase.Continuous,
	:x12     => Multiclass,
	:x13     => Multiclass,
	:x11     => OrderedFactor,
	:x14     => OrderedFactor,
	)

	cols_to_standardize = [1, 4, 5, 8, 10] 
	columns_to_standardize = Matrix(idata[:, cols_to_standardize])

	standardized_columns = DataFrame(StatsBase.standardize(StatsBase.ZScoreTransform, columns_to_standardize, dims=1), :auto)

	o1 = standardized_columns[:,1]
	o4 = standardized_columns[:,2]
	o5 = standardized_columns[:,3]
	o8 = standardized_columns[:,4]
	o10 = standardized_columns[:,5]

	o2 = idata[:,2]
	o3 = idata[:,3]
	o6 = idata[:,6]
	o7 = idata[:,7]
	o9 = idata[:,9]
	o11 = idata[:,11]
	o12 = idata[:,12]
	o13 = idata[:,13]
	o14 = idata[:,14]
	
	idata = DataFrame(hcat(o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14), :auto)
	
	idata = rename(idata, :x1 => :"Age", :x2 => :"Sex", :x3 => :"CP Type", :x4 => :"RBP", :x5 => :"Cholesterol", :x6 => :"BSL", :x7 => :"R ECG", :x8 => :"Max HR", :x9 => :"Ex. Angina", :x10 => :"Oldpeak", :x11 => :"ST", :x12 => :"F. Vessels", :x13 => :"Thal", :x14 => :"Disease");


		idata_copy = coerce(idata_copy, 
	:x1      => MLJBase.Continuous,
	:x2      => Multiclass,
	:x3      => Multiclass,
	:x4      => MLJBase.Continuous,
	:x5      => MLJBase.Continuous,
	:x6      => Multiclass,
	:x7      => Multiclass,
	:x8      => MLJBase.Continuous,
	:x9      => Multiclass,
	:x10     => MLJBase.Continuous,
	:x12     => Multiclass,
	:x13     => Multiclass,
	:x11     => OrderedFactor,
	:x14     => OrderedFactor,
	)
	
	idata_copy = rename(idata_copy, :x1 => :"Age", :x2 => :"Sex", :x3 => :"CP", :x4 => :"RBP", :x5 => :"Chol.", :x6 => :"BSL", :x7 => :"RECG", :x8 => :"Max HR", :x9 => :"Ex. A", :x10 => :"OP", :x11 => :"ST", :x12 => :"F. Ves.", :x13 => :"Thal", :x14 => :"Disease");


end;

# ╔═╡ 497b990e-c80d-4ed6-a825-a26d7a0656e0
begin 
	X1 = idata[:,(1:13)]
	y = idata[:,14]
	X2 = X1[:,[1,4,5,8,10]]
end;

# ╔═╡ df0f0d85-1d65-46f1-aca3-e4b39c055f7c
begin
    Random.seed!(411)
    tr_inds, te_inds = MLJBase.partition(1:MLJ.nrows(X1), 0.7, shuffle=true)
end;

# ╔═╡ 5c566b85-ae97-4214-ada1-1e31cdb5c78d
begin
	X4=Matrix(X2)'
	m = mean(X4, dims=2)
	N = size(X4)[2]
	R = (X4 .- m)*(X4 .- m)' ./ (N-1.0)
	F = eigen(R, sortby = x -> -abs(x))
	W = F.vectors 
	y2 = (W[:,1:5])'*X4
	x = DataFrame(y2[1:3,:]', :auto)
end;

# ╔═╡ e3093ade-4674-460e-b3b4-5bf9703b8f60
ndata = hcat(x, idata_copy);

# ╔═╡ 50408a4a-3730-4b90-bb59-cc3f9d377b5e
tage = md"""
|| 

###### Age:     
""";

# ╔═╡ d6ea4851-4a88-4ea5-8edc-dae2f48340b3
tsex = md"""
||


###### Sex: 
""";

# ╔═╡ 1c17b184-ecd0-409b-84b7-49c61589255f
tcp = md"""
|| 

###### Chest Pain Type: 
""";

# ╔═╡ e992f308-254f-45bc-b980-749ed4467824
tbp = md"""
|| 

###### BP:
""";

# ╔═╡ 7f637209-45d9-4fb3-b901-1ac5a9096925
tchol = md"""
|| 

###### Cholesterol: 
""";

# ╔═╡ 39d678e9-1b45-4792-995f-d40ee67716f3
tbsl = md"""
|| 

###### Fasting BS > 120:
""";

# ╔═╡ 3e7d858b-a416-4339-9e98-88746ecde65d
trecg = md"""
|| 

###### Resting ECG:
""";

# ╔═╡ f885576e-f62d-4675-a642-fe8bbc41e1db
thr = md"""
|| 

###### Max HR: 
""";

# ╔═╡ 47c78483-f9ae-49dd-b7bc-2b50986cca2f
tangina = md"""
|| 

###### Exercsie Angina:
""";

# ╔═╡ 2d2ba8ec-b0ae-4aa2-bb61-c24b8b7ea6b6
top = md"""
|| 

###### Old Peak Value:
""";

# ╔═╡ 42abe919-d5f0-43ea-8ad0-cbcadb770445
tst = md"""
|| 

###### ST value:
""";

# ╔═╡ 2108ee9f-33d8-4320-a632-74033157d382
tfves = md"""
|| 

###### Vessels colored by flourosopy: 
""";

# ╔═╡ dffb51b3-9dfd-4cd9-98a1-83629519ebfc
tthal = md"""
|| 

###### Thal Value:
""";

# ╔═╡ 3219bb09-a9e6-475a-be4a-bd26674c9d74
tdis = md"""
## ||      Heart Disease? 
""";

# ╔═╡ cabc1d1f-c049-46b7-8908-c95869bf351e
bthal = @bind thal MultiCheckBox(["3", "6", "7"])

# ╔═╡ 60098f26-d6ab-4a2c-a434-e0eec66b1fa7
selected_thal = parse.(Int, thal)

# ╔═╡ ca70b7d4-d31e-49f8-b7d1-427159b1d765
tt = 3

# ╔═╡ 5c6bf9b3-791d-4fac-ab6b-31bb276b624f
bdis = @bind dis MultiCheckBox(["Yes", "No"])

# ╔═╡ 77a29201-1a77-4a96-8810-7c3eccc79c83
bfves = @bind fves MultiCheckBox(["0", "1", "2", "3"])

# ╔═╡ 2f93a0d6-7c18-4e00-9ca2-2a615f031592
selected_fves = parse.(Int, fves)

# ╔═╡ 4c42b044-58b6-4d26-bff9-0e4f5f8af6a8
bst = @bind st MultiCheckBox(["1", "2", "3"])

# ╔═╡ fb8ea29d-3cd1-45ec-9740-a87612be5dc9
selected_st = parse.(Int, st)

# ╔═╡ 55c73e97-c04b-48f7-afa2-03a28175de1f
min_op, max_op = extrema(ndata.OP)

# ╔═╡ def03328-7675-4923-9753-4699685d0c54
bop = @bind op_range RangeSlider((min_op:1:max_op))

# ╔═╡ a9d510f7-014c-4c2c-b3fe-85d7fe274658
bangina = @bind angina MultiCheckBox(["Yes", "No"])

# ╔═╡ c71cae95-e9d5-4080-b62f-a791c4f7466d
min_hr, max_hr = extrema(ndata."Max HR");

# ╔═╡ 799c14a2-6f01-455c-b502-5853f7f25278
bhr = @bind hr_range RangeSlider((min_hr:1:max_hr))

# ╔═╡ 39cba144-9cac-44f4-9ffb-7a6460ce76e7
brecg = @bind recg MultiCheckBox(["0", "1", "2"])

# ╔═╡ c70919df-35bc-42d1-a471-866a4f20965f
selected_recg = parse.(Int, recg);

# ╔═╡ 073cc1c7-0433-4fda-a90b-01cbcd1c695f
bbsl = @bind bsl MultiCheckBox(["Yes", "No"])

# ╔═╡ dfbb9c2b-5ea9-4605-a36c-45491ea4732d
min_chol, max_chol = extrema(ndata."Chol.");

# ╔═╡ dad192d8-71b3-42a8-97ca-416673dc46e0
bchol = @bind chol_range RangeSlider((min_chol:1:max_chol))

# ╔═╡ 9de7d77f-dd3d-41e4-8167-37d74eb3cd0d
min_bp, max_bp = extrema(ndata.RBP);

# ╔═╡ 01bd9bc5-944c-4294-9bde-a54b1978e28e
bbp = @bind bp_range RangeSlider((min_bp:1:max_bp))

# ╔═╡ 024d564f-66b4-4620-ba9a-08ecf5fb5d08
bcp = @bind cp MultiCheckBox(["1", "2", "3", "4"])

# ╔═╡ 916fcb0c-4a68-4fce-9d2d-27ef53d1a136
selected_cp = parse.(Int, cp)

# ╔═╡ d5efb1b0-3ff0-4994-9b5d-2400a41792f0
bsex = @bind sex MultiCheckBox(["Male", "Female"])

# ╔═╡ 2d13fdbf-ad16-4a2a-bd66-d7c49c447474
min_age, max_age = extrema(ndata.Age);

# ╔═╡ b5b46055-7b74-4048-8b77-aa30ec0616f9
bage = @bind age_range RangeSlider((min_age:1:max_age))

# ╔═╡ c0d2cd57-833e-4d1e-8890-9a4922a1a301
begin 

	selected_thal
	dis
	selected_fves
	selected_st
	angina
	selected_recg
	bsl
	selected_cp
	sex
	age_range
	bp_range
	chol_range
	hr_range
	op_range

filterd_data = filter(row ->
	        in(row.Thal, selected_thal) &&
	        in(row.Disease, dis) &&
	        in(row."F. Ves.", selected_fves) &&
	        in(row.ST, selected_st) &&
	        in(row."Ex. A", angina) &&
	        in(row.RECG, selected_recg) &&
	        in(row.BSL, bsl) &&
	        in(row.CP, selected_cp) &&
	        in(row.Sex, sex) &&
	        (first(age_range) < row.Age < last(age_range)) &&
	        (first(bp_range) < row.RBP < last(bp_range)) &&
	        (first(chol_range) < row."Chol." < last(chol_range)) &&
	        (first(hr_range) < row."Max HR" < last(hr_range)) &&
	        (first(op_range) < row.OP < last(op_range)),
	    ndata)

end;

# ╔═╡ b6cc7598-0c59-4562-8c10-83b0503c8fb3
begin
	sizes = map(x -> x == "No" ? 3 : 4, filterd_data[!, 17])
	sizes2 = map(x -> x == "No" ? 1 : 3, filterd_data[!, 17])
	tra = map(x -> x == "No" ? 0.5 : 0.9, filterd_data[!, 17])
	colors = map(x -> x == "Yes" ? :orangered : :steelblue, filterd_data[!, 17])
end;

# ╔═╡ 47621f31-3452-4979-a4b7-56c94f65aaa5
begin
	p2 = nothing
	try
	p2 = Plots.scatter(filterd_data[:,11], filterd_data[:,4], color = colors, group=filterd_data[:, 17], ms=sizes, ma = tra, legend=:topright, xlabel = "Max HR", ylabel = "Age",
	xlim = (50, 200), ylim = (20,80), title = "Max HR vs Age")
	catch
    p2 = Plots.scatter([0], [0], color=:black, legend=false, title = "Please Select More Data")
	end
end; 

# ╔═╡ 0b5fcced-a9db-46d9-b96d-ac06a2ac3cdb
begin
	p3 = nothing
	try
	p3 = Plots.scatter(filterd_data[:,7], filterd_data[:,8], group=filterd_data[:, 17], ms=sizes, ma = tra, legend=:topright, ylabel = "Cholesterol", xlabel = "Resting Blood Pressure",
	xlim = (75,200), ylim = (50,650), title = "Cholesterol vs Resting Blood Pressure", color = colors)
	catch
    p3 = Plots.scatter([0], [0], color=:black, legend=false, title = "Please Select More Data")
	end
end;

# ╔═╡ 93c666ea-8fd9-4251-891b-8f7e12b58a33
begin
	plotlyjs()
	p5 = nothing
	try
	p5 = StatsPlots.scatter(
	    filterd_data[:, 1],
	    filterd_data[:, 2],
	    filterd_data[:, 3],
	    group=filterd_data[:, 17],
	    mode="markers", xlabel = "PCA1", ylabel = "PCA2", zlabel = "PCA3", legend = :topleft,
		xlim = (-4,4), ylim = (-3,6), zlim = (-2.5,2),  ms = sizes2, title = "PCA1 vs PCA2 vs PCA3", color = colors)
	catch
    p5 = Plots.scatter([0], [0], color=:black, legend=false, title = "Please Select More Data")
	end
end;

# ╔═╡ 7d88da84-a74b-49be-8801-b29df05f1684
begin
	@info PlutoRunner.currently_running_cell_id
	PlutoUI.ExperimentalLayout.hbox([tsex, bsex, tcp, bcp, tbsl, bbsl, trecg, brecg, tangina, bangina, tst, bst, tfves, bfves, tthal, bthal])
end

# ╔═╡ 1e2a2be9-1ff8-458c-83e4-807224a94976
begin
	@info PlutoRunner.currently_running_cell_id
	PlutoUI.ExperimentalLayout.hbox([tage, bage, tbp, bbp, thr, bhr, top, bop, tchol, bchol, tdis, bdis])
end

# ╔═╡ 7b0a4e45-1182-45d5-8bfb-8e736bd7b531
begin
	@info PlutoRunner.currently_running_cell_id
	PlutoUI.ExperimentalLayout.hbox([p5, p3, p2])
end

# ╔═╡ 03db21c3-0cd8-49fa-be09-7570cb3c6d7b
notebook = PlutoRunner.notebook_id[] |> string

# ╔═╡ 52e313fe-8d8f-4a52-af19-7ef9dd8c7c94
celllist=["7d88da84-a74b-49be-8801-b29df05f1684","1e2a2be9-1ff8-458c-83e4-807224a94976", "7b0a4e45-1182-45d5-8bfb-8e736bd7b531"]

# ╔═╡ 09f47888-d417-49d4-aac6-31c24bc3f424
dash_final_url="http://localhost:1234/edit?" * "id=$notebook&" * join(["isolated_cell_id=$cell" for cell in celllist], "&")

# ╔═╡ 8691dbac-88cb-4b39-875a-58135e5e72ab
@htl("""
<a href="$dash_final_url" style="font_size=20">Click here for the Dashboard</a>
""")

# ╔═╡ Cell order:
# ╟─8691dbac-88cb-4b39-875a-58135e5e72ab
# ╠═783722b3-7252-416f-bf1c-f1c6e89e3b40
# ╠═d5f5b922-5166-4ead-9fa7-4836a9ab6fe6
# ╠═0797d579-8e97-4b79-94f9-b4249609ab4a
# ╠═5ca26dc5-0d67-413d-96d1-6fff9e6639eb
# ╠═b4cdec68-0599-4800-a7dc-1a7cb3cd0fd5
# ╠═497b990e-c80d-4ed6-a825-a26d7a0656e0
# ╠═df0f0d85-1d65-46f1-aca3-e4b39c055f7c
# ╠═42196ee4-99a6-4d54-aa33-ddb7a2591392
# ╠═5c566b85-ae97-4214-ada1-1e31cdb5c78d
# ╠═e3093ade-4674-460e-b3b4-5bf9703b8f60
# ╟─50408a4a-3730-4b90-bb59-cc3f9d377b5e
# ╟─d6ea4851-4a88-4ea5-8edc-dae2f48340b3
# ╟─1c17b184-ecd0-409b-84b7-49c61589255f
# ╟─e992f308-254f-45bc-b980-749ed4467824
# ╟─7f637209-45d9-4fb3-b901-1ac5a9096925
# ╟─39d678e9-1b45-4792-995f-d40ee67716f3
# ╟─3e7d858b-a416-4339-9e98-88746ecde65d
# ╟─f885576e-f62d-4675-a642-fe8bbc41e1db
# ╟─47c78483-f9ae-49dd-b7bc-2b50986cca2f
# ╟─2d2ba8ec-b0ae-4aa2-bb61-c24b8b7ea6b6
# ╟─42abe919-d5f0-43ea-8ad0-cbcadb770445
# ╟─2108ee9f-33d8-4320-a632-74033157d382
# ╟─dffb51b3-9dfd-4cd9-98a1-83629519ebfc
# ╟─3219bb09-a9e6-475a-be4a-bd26674c9d74
# ╠═cabc1d1f-c049-46b7-8908-c95869bf351e
# ╠═60098f26-d6ab-4a2c-a434-e0eec66b1fa7
# ╠═ca70b7d4-d31e-49f8-b7d1-427159b1d765
# ╠═5c6bf9b3-791d-4fac-ab6b-31bb276b624f
# ╠═77a29201-1a77-4a96-8810-7c3eccc79c83
# ╠═2f93a0d6-7c18-4e00-9ca2-2a615f031592
# ╠═4c42b044-58b6-4d26-bff9-0e4f5f8af6a8
# ╠═fb8ea29d-3cd1-45ec-9740-a87612be5dc9
# ╠═55c73e97-c04b-48f7-afa2-03a28175de1f
# ╠═def03328-7675-4923-9753-4699685d0c54
# ╠═a9d510f7-014c-4c2c-b3fe-85d7fe274658
# ╠═c71cae95-e9d5-4080-b62f-a791c4f7466d
# ╠═799c14a2-6f01-455c-b502-5853f7f25278
# ╠═39cba144-9cac-44f4-9ffb-7a6460ce76e7
# ╠═c70919df-35bc-42d1-a471-866a4f20965f
# ╠═073cc1c7-0433-4fda-a90b-01cbcd1c695f
# ╠═dfbb9c2b-5ea9-4605-a36c-45491ea4732d
# ╠═dad192d8-71b3-42a8-97ca-416673dc46e0
# ╠═9de7d77f-dd3d-41e4-8167-37d74eb3cd0d
# ╠═01bd9bc5-944c-4294-9bde-a54b1978e28e
# ╠═024d564f-66b4-4620-ba9a-08ecf5fb5d08
# ╠═916fcb0c-4a68-4fce-9d2d-27ef53d1a136
# ╠═d5efb1b0-3ff0-4994-9b5d-2400a41792f0
# ╠═2d13fdbf-ad16-4a2a-bd66-d7c49c447474
# ╠═b5b46055-7b74-4048-8b77-aa30ec0616f9
# ╠═c0d2cd57-833e-4d1e-8890-9a4922a1a301
# ╠═b6cc7598-0c59-4562-8c10-83b0503c8fb3
# ╠═47621f31-3452-4979-a4b7-56c94f65aaa5
# ╠═0b5fcced-a9db-46d9-b96d-ac06a2ac3cdb
# ╠═93c666ea-8fd9-4251-891b-8f7e12b58a33
# ╠═7d88da84-a74b-49be-8801-b29df05f1684
# ╠═1e2a2be9-1ff8-458c-83e4-807224a94976
# ╠═7b0a4e45-1182-45d5-8bfb-8e736bd7b531
# ╠═03db21c3-0cd8-49fa-be09-7570cb3c6d7b
# ╠═52e313fe-8d8f-4a52-af19-7ef9dd8c7c94
# ╠═09f47888-d417-49d4-aac6-31c24bc3f424
